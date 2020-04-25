#include <tensorflow/c/c_api.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "absl/synchronization/mutex.h"

// adapted from
// https://stackoverflow.com/questions/44378764/hello-tensorflow-using-the-c-api
// and
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
  // Create a absl::Mutex here.
  absl::Mutex m;
  absl::MutexLock lock(&m);
  // load graph
  TF_Status* status = TF_NewStatus();
  TF_Buffer* buffer = ReadBufferFromFile("load_model.pb");
  auto graph = TF_NewGraph();
  auto opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);
  // Open session
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);
  // Set tensors.
  int size = 100;
  int64_t dims[2] = {1, size};
  // input a.
  TF_Output input_a;
  input_a.oper = TF_GraphOperationByName(graph, "input_a");
  input_a.index = 0;
  TF_Tensor* intput_a_tensor =
      TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * size);
  for (int i = 0; i != size; ++i) {
    *(static_cast<float*>(TF_TensorData(intput_a_tensor)) + i) = 2 * i;
  }
  // input b.
  TF_Output input_b;
  input_b.oper = TF_GraphOperationByName(graph, "input_b");
  input_b.index = 0;
  TF_Tensor* intput_b_tensor =
      TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * size);
  for (int i = 0; i != size; ++i) {
    *(static_cast<float*>(TF_TensorData(intput_b_tensor)) + i) = -i;
  }
  // output. ouput = input_a + input_b.
  TF_Output output;
  output.oper = TF_GraphOperationByName(graph, "result");
  output.index = 0;
  TF_Tensor* output_tensor;

  std::vector<TF_Output> inputs = {input_a, input_b};
  std::vector<TF_Tensor*> input_tensors = {intput_a_tensor, intput_b_tensor};

  std::cout << "Session starting" << std::endl;
  TF_SessionRun(session, nullptr, inputs.data(), input_tensors.data(),
                static_cast<int>(input_tensors.size()),  // Inputs
                &output, &output_tensor, 1,              // Outputs
                nullptr, 0,                              // Operations
                nullptr, status);
  std::cout << "Session finisehd" << std::endl;

  for (int i = 0; i != size; ++i) {
    std::cout << *(static_cast<float*>(TF_TensorData(output_tensor)) + i)
              << " ";
  }
  std::cout << std::endl;

  TF_DeleteTensor(intput_a_tensor);
  TF_DeleteTensor(intput_b_tensor);
  TF_DeleteTensor(output_tensor);
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);

  return 0;
}