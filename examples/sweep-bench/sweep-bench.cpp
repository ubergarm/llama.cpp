#include "common.h"
#include "arg.h"
#include "ggml.h"
#include "llama.h"
#include "common.h"
//#include "llama-vocab.h"
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

    std::vector<char*> args;
    args.reserve(argc);
    args.push_back(argv[0]);

    bool sweep_bench_output_jsonl = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[1]};
        if (arg == "--output-format") {
            bool invalid_arg = false;
            if (i < argc-1) {
                arg = argv[++i];
                if (arg == "jsonl") sweep_bench_output_jsonl = true;
                else if (arg == "md") sweep_bench_output_jsonl = false;
                else invalid_arg = true;
            } else {
                invalid_arg = true;
            }
            if (invalid_arg) {
                LOG("Invalid arg"); return 1;
            }
        } else {
            args.push_back(argv[i]);
        }
    }

    common_params params;
    if (!common_params_parse(args.size(), args.data(), params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    common_init();

    //gpt_params params;

    //if (!gpt_params_parse(argc, argv, params)) {
    //    print_usage(argc, argv);
    //    return 1;
    //}

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    //llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model_params model_params = common_model_params_to_llama(params);

    //llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    //llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context_params ctx_params = common_context_params_to_llama(params);

    //llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    auto * mem = llama_get_memory(ctx);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const unsigned int n_kv_max = llama_n_ctx(ctx);


    auto vocab   = llama_model_get_vocab(model);
    auto n_vocab = llama_vocab_n_tokens(vocab);
    auto bos     = llama_vocab_bos(vocab);

    //const llama_vocab * vocab = llama_get_vocab(ctx);
    //llama_token bos = llama_token_bos_impl(*vocab);
    //llama_token eos = llama_token_eos_impl(*vocab);

    //const unsigned int n_vocab  = llama_n_vocab(model);

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
        LOG_INF("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn_type = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn_type, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG_INF("\n");
        LOG_INF("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
        LOG_INF("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // warm up
    {
        common_batch_add(batch, bos, 0, { 0 }, false);
        //llama_batch_add(batch, bos, 0, { 0 }, false);

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    // Adapted into mainline from original PR: https://github.com/ikawrakow/ik_llama.cpp/pull/375
    //if (params.batch_warmup) {
    if (true) {
        // clean up KV cache after generation
        // llama_kv_self_clear(ctx);
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
    }

    common_batch_clear(batch);
    //llama_batch_clear(batch);
    //llama_kv_self_clear(ctx);
    llama_memory_clear(mem, true);

    for (unsigned int n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        // clean up KV cache before generation
        //llama_kv_self_seq_rm(ctx, 0, n_kv, -1);
        llama_memory_seq_rm(mem, 0, n_kv, -1);

        // first measure token generation performance at this context size
        const auto t_tg_start = ggml_time_us();

        for (unsigned int i = 0; i < tg; ++i) {
            common_batch_clear(batch);
            common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);
            //llama_batch_clear(batch);
            //llama_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);

            if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                LOG_INF("%s: llama_decode() failed\n", __func__);
                return 1;
            }
        }

        const auto t_tg_end = ggml_time_us();

        // clean up KV cache after generation
        //llama_kv_self_seq_rm(ctx, 0, n_kv, -1);
        llama_memory_seq_rm(mem, 0, n_kv, -1);

        // prepare batch of pp size for prompt processing performance measurement
        common_batch_clear(batch);
        //llama_batch_clear(batch);

        for (unsigned int i = 0; i < pp; ++i) {
            common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
            //llama_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        // measure prompt processing performance
        const auto t_pp_start = ggml_time_us();

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        const auto t_pp_end = ggml_time_us();

        // calculate and print metrics
        const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
        const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;

        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        if(sweep_bench_output_jsonl) {
            LOG_INF(
                "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn_type\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f }\n",
                n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn_type, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg
            );
        } else {
            LOG_INF("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
        }
    }

    llama_batch_free(batch);

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
