from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


@lru_cache
def get_client():
    return InferenceServerClient(url="localhost:8500")


def main():
    triton_client = get_client()

    first_img = np.load("./test/l1.npy")
    second_img = np.load("./test/l2.npy")
    third_img = np.load("./test/l3.npy")
    fourth_img = np.load("./test/l4.npy")

    print(first_img.shape)
    print(second_img.shape)
    print(third_img.shape)
    print(fourth_img.shape)

    inputs = []
    outputs = []
    inputs.append(InferInput("input", [1, 3, 96, 128], "FP32"))
    inputs[0].set_data_from_numpy(first_img)

    inputs.append(InferInput("onnx::Add_1", [1, 3, 192, 256], "FP32"))
    inputs[1].set_data_from_numpy(second_img)

    inputs.append(InferInput("onnx::Add_2", [1, 3, 384, 512], "FP32"))
    inputs[2].set_data_from_numpy(third_img)

    inputs.append(InferInput("onnx::Add_3", [1, 3, 768, 1024], "FP32"))
    inputs[3].set_data_from_numpy(fourth_img)

    outputs.append(InferRequestedOutput("output"))
    outputs.append(InferRequestedOutput("onnx::Add_872"))
    outputs.append(InferRequestedOutput("onnx::Add_1205"))
    outputs.append(InferRequestedOutput("1467"))

    results = triton_client.infer("onnx-mnist", inputs, outputs=outputs)

    first_out = np.squeeze(results.as_numpy("output"))
    second_out = np.squeeze(results.as_numpy("onnx::Add_872"))
    third_out = np.squeeze(results.as_numpy("onnx::Add_1205"))
    fourth_out = np.squeeze(results.as_numpy("1467"))
    np.save("./test/o1", first_out)
    np.save("./test/o2", second_out)
    np.save("./test/o3", third_out)
    np.save("./test/o4", fourth_out)


if __name__ == "__main__":
    main()
