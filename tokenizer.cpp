#include <iostream>
#include "json/single_include/nlohmann/json.hpp"
#include <fstream>
#include <chrono>
#include <vector>
#include <map>



#include <string>
using json = nlohmann::json;

int main() {
    auto total_start = std::chrono::high_resolution_clock::now();

    // Measure file opening time
    auto open_start = std::chrono::high_resolution_clock::now();
    std::ifstream file("tokenizer/tokenizer.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return 1;
    }
    auto open_end = std::chrono::high_resolution_clock::now();

    // Measure JSON parsing time
    auto parse_start = std::chrono::high_resolution_clock::now();
    json tokenizer_data;
    file >> tokenizer_data;
    auto parse_end = std::chrono::high_resolution_clock::now();
    
    file.close();
    std::map<int, std::string> id2Token;
  
    for(auto &[key,item]:tokenizer_data["model"]["vocab"].items()){
        id2Token[item]=key;
    }

    // Measure token lookup time
    auto lookup_start = std::chrono::high_resolution_clock::now();
    std::string prompt=" hello there missfortune ! ";
    std::vector<std::string> tokensList;
    tokensList.push_back("[CLS]");
    std::vector<int>idsList;
    idsList.push_back(tokenizer_data["model"]["vocab"]["[CLS]"]);
    std::string ch="";
    for(int i=0;i<prompt.size();i++){
        if(prompt[i]==' ') {if(ch!="")
            
            {
            auto it=tokenizer_data["model"]["vocab"].find(ch);
            if(it!=tokenizer_data["model"]["vocab"].end()){
            idsList.push_back(tokenizer_data["model"]["vocab"][ch]);
            tokensList.push_back(ch);
            }
            else{
                std::string subch="";
                bool First=true;
                for(int j=0;j<ch.size();j++){
                    if(First){
                    if(!tokenizer_data["model"]["vocab"].contains(subch+ch[j])){
                        
                        idsList.push_back(tokenizer_data["model"]["vocab"][subch]);tokensList.push_back(subch); First=false;subch="";}
                       
                    }
                    else {
                        if(!tokenizer_data["model"]["vocab"].contains("##"+subch+ch[j])){
                        
                        idsList.push_back(tokenizer_data["model"]["vocab"]["##"+subch]); tokensList.push_back("##"+subch);subch="";}
                    }
                    subch+=ch[j];
                }
                if(tokenizer_data["model"]["vocab"].contains("##"+subch)){
                    if(First){
                        idsList.push_back(tokenizer_data["model"]["vocab"]["##"+subch]);tokensList.push_back("##"+subch); First=false;}
                        else {idsList.push_back(tokenizer_data["model"]["vocab"]["##"+subch]); tokensList.push_back("##"+subch);}
                        subch="";
                }
                if(First==true){
                    idsList.push_back(-1);
                }
            }

           } ch="";}
        else{
            ch+=prompt[i];
        }
    }
    if(ch!="") {tokensList.push_back(ch);idsList.push_back(tokenizer_data["model"]["vocab"][ch]);}
    tokensList.push_back("[SEP]");
    idsList.push_back(tokenizer_data["model"]["vocab"]["[SEP]"]);

    for(auto x:idsList) std::cout<<x<<" ";
    std::cout<<std::endl;
    
    for(auto x:tokensList) std::cout<<x<<" ";
    std::cout<<std::endl;
    std::string res="";
    std::vector<std::string> Decoded;
    for(int i=0;i<idsList.size();i++){
        if(id2Token[idsList[i]][0]!='#'){
            Decoded.push_back(id2Token[idsList[i]]);

        }
        else{
            Decoded[Decoded.size()-1]+=id2Token[idsList[i]].substr(2);
        }
    }
    
    for(auto x:Decoded) std::cout<<x<<" ";
    std::cout<<std::endl;
    

    
  /*   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BERT");
    
    // 2. Configure session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    try {

        // 3. Load the ONNX model
        Ort::Session session(env, "model.onnx", session_options);

        // 4. Get model metadata
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of inputs: " << num_input_nodes << std::endl;

        // 5. Prepare sample input (modify according to your model's requirements)
        std::vector<int64_t> input_shape = {1, 128};  // Example shape
        std::vector<float> input_values(128, 0.5f);   // Sample data

        // 6. Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 7. Prepare input names (modify according to your model)
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        // 8. Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // 9. Process output
        if (output_tensors.size() > 0 && output_tensors[0].IsTensor()) {
            float* output = output_tensors[0].GetTensorMutableData<float>();
            std::cout << "Inference successful! First output value: " << output[0] << std::endl;
        }
 */
        

        // input_ids
        

        

   


    auto lookup_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    // Print timings
    std::cout << "File Open Time: " << std::chrono::duration<double>(open_end - open_start).count() << " sec\n";
    std::cout << "JSON Parse Time: " << std::chrono::duration<double>(parse_end - parse_start).count() << " sec\n";
    std::cout << "Token Lookup Time: " << std::chrono::duration<double>(lookup_end - lookup_start).count() << " sec\n";
    std::cout << "Total Execution Time: " << std::chrono::duration<double>(total_end - total_start).count() << " sec\n";

    return 0;
}
