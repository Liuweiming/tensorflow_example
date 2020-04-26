#include <tensorflow/c/c_api.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "absl/synchronization/mutex.h"

// adapted from
// https://github.com/Neargye/hello_tf_c_api

static void DeallocateBuffer(void* data, size_t) { std::free(data); }

static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::ifstream f(file, std::ios::binary);
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  if (f.seekg(0, std::ios::end).fail()) {
    return nullptr;
  }
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail()) {
    return nullptr;
  }

  if (fsize <= 0) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(fsize));
  if (f.read(data, fsize).fail()) {
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;
  f.close();
  return buf;
}

int main(int argc, char** argv) {
  // Create absl::Mutex here.
  // absl::Mutex m;

  // load graph
  TF_Status* status = TF_NewStatus();
  TF_Buffer* buffer = ReadBufferFromFile("training.pb");
  auto graph = TF_NewGraph();
  auto opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  // Open session
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  // Get init and train operations.
  TF_Operation* init = TF_GraphOperationByName(graph, "init");
  TF_Operation* train = TF_GraphOperationByName(graph, "train");

  // Run init.
  std::cout << "Session starting: init" << std::endl;
  TF_SessionRun(session, nullptr, nullptr, nullptr, 0,  // Inputs
                nullptr, nullptr, 0,                    // Outputs
                &init, 1,                               // Operations
                nullptr, status);
  std::cout << "Session ending: init" << std::endl;

  // Set tensors.
  int input_dim = 100;
  int output_dim = 3;
  int64_t input_dims[2] = {1, input_dim};
  int64_t output_dims[2] = {1, output_dim};
  // input.
  TF_Output input;
  input.oper = TF_GraphOperationByName(graph, "input");
  input.index = 0;
  TF_Tensor* input_tensor =
      TF_AllocateTensor(TF_FLOAT, input_dims, 2, sizeof(float) * input_dim);
  for (int i = 0; i != input_dim; ++i) {
    *(static_cast<float*>(TF_TensorData(input_tensor)) + i) =
        (float)i / input_dim;
  }
  // target.
  TF_Output target;
  target.oper = TF_GraphOperationByName(graph, "target");
  target.index = 0;
  TF_Tensor* target_tensor =
      TF_AllocateTensor(TF_FLOAT, output_dims, 2, sizeof(float) * output_dim);
  for (int i = 0; i != output_dim; ++i) {
    *(static_cast<float*>(TF_TensorData(target_tensor)) + i) = i;
  }
  // output.
  TF_Output output;
  output.oper = TF_GraphOperationByName(graph, "output");
  output.index = 0;
  TF_Tensor* output_tensor;

  // Inference.
  std::cout << "Session starting: output" << std::endl;
  TF_SessionRun(session, nullptr, &input, &input_tensor, 1,  // Inputs
                &output, &output_tensor, 1,                  // Outputs
                nullptr, 0,                                  // Operations
                nullptr, status);
  std::cout << "Session ending: output" << std::endl;
  std::cout << "before training, ouput = [";
  for (int i = 0; i != output_dim; ++i) {
    std::cout << *(static_cast<float*>(TF_TensorData(output_tensor)) + i)
              << " ";
  }
  std::cout << "]" << std::endl;

  // Training.
  std::vector<TF_Output> inputs = {input, target};
  std::vector<TF_Tensor*> input_tensors = {input_tensor, target_tensor};
  std::cout << "Session starting: train" << std::endl;
  for (int iter = 0; iter != 100; ++iter) {
    TF_DeleteTensor(output_tensor);
    TF_SessionRun(session, nullptr, inputs.data(), input_tensors.data(),
                  static_cast<int>(input_tensors.size()),  // Inputs
                  &output, &output_tensor, 1,              // Outputs
                  &train, 1,                               // Operations
                  nullptr, status);
  }
  std::cout << "Session ending: train" << std::endl;
  std::cout << "after training, ouput = [";
  for (int i = 0; i != output_dim; ++i) {
    std::cout << *(static_cast<float*>(TF_TensorData(output_tensor)) + i)
              << " ";
  }
  std::cout << "]" << std::endl;

  TF_DeleteTensor(input_tensor);
  TF_DeleteTensor(target_tensor);
  TF_DeleteTensor(output_tensor);
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);

  return 0;
}