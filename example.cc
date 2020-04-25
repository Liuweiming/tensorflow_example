#include <tensorflow/c/c_api.h>
#include <cstring>
#include <iostream>
#include "absl/synchronization/mutex.h"

// adapted from
// https://stackoverflow.com/questions/44378764/hello-tensorflow-using-the-c-api

int main(int argc, char** argv) {
  // Create a absl::Mutex here.
  absl::Mutex m;
  absl::MutexLock lock(&m);

  TF_Graph* graph = TF_NewGraph();
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Status* status = TF_NewStatus();
  TF_Session* session = TF_NewSession(graph, options, status);
  char hello[] = "Hello TensorFlow!";
  TF_Tensor* tensor = TF_AllocateTensor(
      TF_STRING, 0, 0, 8 + TF_StringEncodedSize(strlen(hello)));
  TF_Tensor* tensorOutput;
  TF_OperationDescription* operationDescription =
      TF_NewOperation(graph, "Const", "hello");
  TF_Operation* operation;
  struct TF_Output output;

  TF_StringEncode(hello, strlen(hello), 8 + (char*)TF_TensorData(tensor),
                  TF_StringEncodedSize(strlen(hello)), status);
  memset(TF_TensorData(tensor), 0, 8);
  TF_SetAttrTensor(operationDescription, "value", tensor, status);
  TF_SetAttrType(operationDescription, "dtype", TF_TensorType(tensor));
  operation = TF_FinishOperation(operationDescription, status);

  output.oper = operation;
  output.index = 0;
  std::cout << "Session starting" << std::endl;
  TF_SessionRun(session, 0, 0, 0, 0,        // Inputs
                &output, &tensorOutput, 1,  // Outputs
                &operation, 1,              // Operations
                0, status);
  std::cout << "Session finisehd" << std::endl;
  std::cout << std::string((char*)TF_TensorData(tensorOutput) + 9) << std::endl;

  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
  TF_DeleteSessionOptions(options);

  return 0;
}