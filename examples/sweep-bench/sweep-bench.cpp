#include "arg.h"
#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "log.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 8192 -b 2048 -ub 512\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    // Parse output-format arg first (sweep-bench specific)
    std::vector<char*> args;
    args.reserve(argc);
    args.push_back(argv[0]);

    bool sweep_bench_output_jsonl = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[i]};
        if (arg == "--output-format") {
            if (i < argc-1) {
                arg = argv[++i];
                if (arg == "jsonl") sweep_bench_output_jsonl = true;
                else if (arg == "md") sweep_bench_output_jsonl = false;
            }
        } else {
            args.push_back(argv[i]);
        }
    }

    // Setup logging and parse common params
    common_init();

    common_params params;
    if (!common_params_parse(args.size(), args.data(), params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    // Initialize using common pattern
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s: failed to init\n", __func__);
        return 1;
    }

    auto * mem = llama_get_memory(ctx);
    auto ctx_params = common_context_params_to_llama(params);

    const unsigned int n_kv_max = llama_n_ctx(ctx);

    auto vocab   = llama_model_get_vocab(model);
    auto n_vocab = llama_vocab_n_tokens(vocab);
    auto bos     = llama_vocab_bos(vocab);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_INF("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            llama_synchronize(ctx);
        }

        return true;
    };

    const unsigned int pp = params.n_ubatch;
    const unsigned int tg = params.n_ubatch / 4;

    if (!sweep_bench_output_jsonl) {
        LOG_INF("\n");
        LOG_INF("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn_type, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG_INF("\n");
        LOG_INF("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
        LOG_INF("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // For recurrent and hybrid models, we need to use checkpoints because
    // partial removal of tokens is not supported with recurrent/hybrid state
    const bool is_recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);

    // warm up
    {
        common_batch_add(batch, bos, 0, { 0 }, false);

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    // Adapted into mainline from original PR: https://github.com/ikawrakow/ik_llama.cpp/pull/375

    // clean up KV cache after generation
    llama_memory_clear(mem, true);

    // prepare batch of pp size for prompt processing performance measurement
    common_batch_clear(batch);

    for (unsigned int i = 0; i < (unsigned int)params.n_ubatch; ++i) {
        common_batch_add(batch, std::rand() % n_vocab, i, { 0 }, false);
    }

    if (!decode_helper(ctx, batch, ctx_params.n_ubatch)) {
        LOG_INF("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    common_batch_clear(batch);
    llama_memory_clear(mem, true);

    // For recurrent models, we use state checkpoints to avoid rebuilding context from scratch
    std::vector<uint8_t> checkpoint_data;

    for (unsigned int n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        // Prepare context at exactly n_kv tokens before TG
        if (is_recurrent) {
            if (n_kv == 0) {
                // For n_kv=0, ensure empty context
                llama_memory_clear(mem, true);
            } else if (!checkpoint_data.empty()) {
                // Restore checkpoint representing state at n_kv tokens
                const size_t ret = llama_state_seq_set_data_ext(ctx, checkpoint_data.data(), checkpoint_data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (ret != checkpoint_data.size()) {
                    LOG_INF("%s: failed to restore checkpoint, falling back to rebuild\n", __func__);
                    checkpoint_data.clear();
                    llama_memory_clear(mem, true);
                    for (unsigned int pos = 0; pos < n_kv; pos += params.n_batch) {
                        const unsigned int build_tokens = std::min((unsigned int)params.n_batch, n_kv - pos);
                        common_batch_clear(batch);
                        for (unsigned int j = 0; j < build_tokens; ++j) {
                            common_batch_add(batch, std::rand() % n_vocab, pos + j, { 0 }, false);
                        }
                        if (!decode_helper(ctx, batch, params.n_batch)) {
                            LOG_INF("%s: llama_decode() failed to build context\n", __func__);
                            return 1;
                        }
                    }
                }
            } else {
                // No checkpoint yet, build from scratch (first iteration after warmup)
                llama_memory_clear(mem, true);
                for (unsigned int pos = 0; pos < n_kv; pos += params.n_batch) {
                    const unsigned int build_tokens = std::min((unsigned int)params.n_batch, n_kv - pos);
                    common_batch_clear(batch);
                    for (unsigned int j = 0; j < build_tokens; ++j) {
                        common_batch_add(batch, std::rand() % n_vocab, pos + j, { 0 }, false);
                    }
                    if (!decode_helper(ctx, batch, params.n_batch)) {
                        LOG_INF("%s: llama_decode() failed to build context\n", __func__);
                        return 1;
                    }
                }
            }
        } else {
            // For transformer models, simply remove tokens beyond n_kv
            llama_memory_seq_rm(mem, 0, n_kv, -1);
        }

        // first measure token generation performance at this context size
        const auto t_tg_start = ggml_time_us();

        for (unsigned int i = 0; i < tg; ++i) {
            common_batch_clear(batch);
            common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);

            if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                LOG_INF("%s: llama_decode() failed\n", __func__);
                return 1;
            }
        }

        const auto t_tg_end = ggml_time_us();

        // After TG, need to return to exactly n_kv tokens before PP
        if (is_recurrent) {
            // Restore checkpoint to revert TG tokens
            if (!checkpoint_data.empty()) {
                const size_t ret = llama_state_seq_set_data_ext(ctx, checkpoint_data.data(), checkpoint_data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (ret != checkpoint_data.size()) {
                    LOG_INF("%s: failed to restore checkpoint after TG\n", __func__);
                    // If restore fails, we cannot measure PP correctly
                    return 1;
                }
            } else {
                // This shouldn't happen: TG ran but no checkpoint from before?
                // Fallback: clear everything (only valid if n_kv==0)
                if (n_kv != 0) {
                    LOG_INF("%s: cannot recover from missing checkpoint at n_kv=%d\n", __func__, n_kv);
                    return 1;
                }
                llama_memory_clear(mem, true);
            }
        } else {
            // For transformer models, remove TG tokens
            llama_memory_seq_rm(mem, 0, n_kv, -1);
        }

        // prepare batch of pp size for prompt processing performance measurement
        common_batch_clear(batch);

        for (unsigned int i = 0; i < pp; ++i) {
            common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        // measure prompt processing performance
        const auto t_pp_start = ggml_time_us();

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        const auto t_pp_end = ggml_time_us();

        // Save checkpoint for next iteration (recurrent models only)
        if (is_recurrent) {
            const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            checkpoint_data.resize(checkpoint_size);
            const size_t written = llama_state_seq_get_data_ext(ctx, checkpoint_data.data(), checkpoint_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (written != checkpoint_size) {
                LOG_INF("%s: failed to save checkpoint\n", __func__);
                checkpoint_data.clear();
            }
        }

        // calculate and print metrics
        const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
        const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;

        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        if(sweep_bench_output_jsonl) {
            LOG_INF(
                "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f }\n",
                n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn_type, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg
            );
        } else {
            LOG_INF("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
        }
    }

    llama_batch_free(batch);

    return 0;
}