#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <typeinfo>  
#include <chrono>

#include <random>
#include <limits>

#include <iomanip>  
using namespace std;






void top_p_sampling(std::vector<float>& logits, float top_p = 0.95f) {
   
    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    
    std::sort(indices.begin(), indices.end(),
        [&logits](size_t a, size_t b) { return logits[a] > logits[b]; });

    // Convert logits to probabilities using softmax
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (auto& p : probs) p /= sum_exp;

    
    std::vector<float> sorted_probs;
    for (auto i : indices) {
        sorted_probs.push_back(probs[i]);
    }

    // cumulative probabilities
    std::vector<float> cum_probs(sorted_probs.size());
    std::partial_sum(sorted_probs.begin(), sorted_probs.end(), cum_probs.begin());

    size_t cutoff = 0;
    for (; cutoff < cum_probs.size(); ++cutoff) {
        if (cum_probs[cutoff] > top_p) {
            break;
        }
    }

    // Create mask for tokens to keep
    std::vector<bool> mask(logits.size(), false);
    for (size_t i = 0; i <= cutoff && i < indices.size(); ++i) {
        mask[indices[i]] = true;
    }

    // Apply mask - set unwanted logits to -inf
    float min_float = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!mask[i]) {
            logits[i] = min_float;
        }
    }
}

// Function to sample from probability distribution
int sample_from_probs(const std::vector<float>& probs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

// Main generation function
std::vector<int64_t> generate_text(
    Ort::Session& session,
    const std::vector<int64_t>& initial_input,
    size_t vocab_size,
    size_t max_length = 5,
    float top_p = 0.9f,
    float repetition_penalty = 1.1f,
    int eos_token_id = -1) {

    std::vector<int64_t> input_ids = initial_input;
    

    // Memory information and session setup
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name.get() };
    const char* output_names[] = { output_name.get() };

    // Generate sequence
    for (size_t i = 0; i < max_length; ++i) {
        // Prepare input tensor
        std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(input_ids.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Get logits from output
       // float* logits = output_tensors[0].GetTensorMutableData<float>();

        //size_t num_elements = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        //cout << num_elements << endl;
        
        //std::vector<float> LastLogits;
       // for (size_t c = (i + 1) * vocab_size; c < num_elements; c++) {
         //   LastLogits.push_back(logits[c] / repetition_penalty); 
       // }


        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        float* logits_ptr = output_tensors[0].GetTensorMutableData<float>();

        // Calculate index of the last token's logits
        // For shape [batch_size, sequence_length, vocab_size], we want the last sequence element
        size_t last_token_index = input_ids.size() - 1;
        size_t logits_offset = last_token_index * vocab_size;

        // Extract logits for the last token
        std::vector<float> LastLogits(vocab_size);
        for (size_t v = 0; v < vocab_size; v++) {
            LastLogits[v] = logits_ptr[logits_offset + v] / repetition_penalty;
        }



        // Apply top-p sampling
        top_p_sampling(LastLogits, top_p);

        // Convert logits to probabilities
        std::vector<float> probs(LastLogits.size());
        

        //probs.resize(LastLogits.size());  // Ensure probs is the right size
        float max_logit = *std::max_element(LastLogits.begin(), LastLogits.end());

        // Step 4: Compute the sum of exponentials for numerical stability
        double sum_exp = 0.0;
        for (size_t i = 0; i < LastLogits.size(); ++i) {
            // Use double precision for exponentiation for better accuracy
            sum_exp += std::exp(static_cast<double>(LastLogits[i]) - static_cast<double>(max_logit));
        }

        // Step 5: Normalize to get probabilities
        for (size_t i = 0; i < LastLogits.size(); ++i) {
            // Normalize probabilities and convert back to float
            probs[i] = static_cast<float>(std::exp(static_cast<double>(LastLogits[i]) - static_cast<double>(max_logit)) / sum_exp);
        }



      
        //cout << vocab_size << endl;
        //cout << std::fixed << std::setprecision(30) << probs[i] << std::endl;
        // Sample next token
        int next_token_id = sample_from_probs(probs);
        input_ids.push_back(next_token_id);
        //cout << next_token_id << endl;
        // Break if EOS token is generated
        if (next_token_id == 50256) {
            break;
        }
    }
    return input_ids;
}


int main() {



    auto model_path = L"C:\\Users\\ymahjoub\\source\\repos\\Project1\\Project1\\assets\\model.onnx";
        
        Ort::Env env;
        Ort::RunOptions runoptions;
        Ort::Session session(nullptr);
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model_path, session_options);
        std::cout << "success" << endl;
    

        const size_t vocab_size = 50257;

        std::vector<int64_t> initial_input = { 464, 6193, 1909,  318 };
        auto start_time = std::chrono::high_resolution_clock::now();
         auto generated_ids = generate_text(session, initial_input, vocab_size);
         auto end_time = std::chrono::high_resolution_clock::now();

         // Calculate duration
         std::chrono::duration<double> elapsed_seconds = end_time - start_time;
          std::cout << "\nExecution time: " << elapsed_seconds.count() << " seconds\n";
    // Decode and print results
    for (auto id : generated_ids) {
        std::cout << id << " ";
    }
       


    return 0;
}