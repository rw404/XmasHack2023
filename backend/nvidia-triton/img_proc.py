import cv2
import numpy as np
import torch

# from model import MSPEC_Net

IN_SIZE = 512


def decomposition(img, level_num=4):
    G_list = []
    L_list = []
    G_list.append(img)
    for i in range(level_num - 1):
        G_list.append(cv2.pyrDown(G_list[i]))
    for j in range(level_num - 1):
        L_list.append(
            G_list[j]
            - cv2.pyrUp(G_list[j + 1], dstsize=(G_list[j].shape[1], G_list[j].shape[0]))
        )
    L_list.append(G_list[level_num - 1])
    G_list.reverse()
    L_list.reverse()
    return G_list, L_list


def img_to_list(img):
    maxsize = max([img.shape[0], img.shape[1]])
    scale_ratio = IN_SIZE / maxsize
    im_low = cv2.resize(
        img, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC
    )
    top_pad, left_pad = IN_SIZE - im_low.shape[0], IN_SIZE - im_low.shape[1]
    img = cv2.copyMakeBorder(im_low, top_pad, 0, left_pad, 0, cv2.BORDER_DEFAULT)

    if img.dtype == "uint8":
        img = img / 255
    _, L_list = decomposition(img)

    L_list = [
        torch.from_numpy(data).float().permute(2, 0, 1).unsqueeze(0).cuda()
        for data in L_list
    ]

    return L_list, (top_pad, left_pad)


def list_to_img(Y_list, pad, shape):
    out = Y_list[-1].squeeze().permute(1, 2, 0).detach().cpu().numpy()
    out = out[pad[0] :, pad[1] :, :]

    out = cv2.resize(out, (shape[1], shape[0])) * 255
    out = out.clip(0, 255).astype(np.uint8)

    return out


# MSPEC_net = MSPEC_Net().cuda()
# weights = torch.load('./snapshots/MSPECnet_woadv.pth')
# for w in list(weights.keys()):
#     weights[w[7:]] = weights.pop(w)
# MSPEC_net.load_state_dict(weights)
# MSPEC_net.eval()

# img = cv2.imread("tests/IMG_0647.png")
# l, pad = img_to_list(img)
# l = MSPEC_net(l)
# img = list_to_img(l, pad, img.shape)
# cv2.imwrite("res/bue.png", img)
