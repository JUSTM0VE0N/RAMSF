import numpy as np
from numpy.linalg import norm
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage.filters import sobel, convolve
from scipy.stats import pearsonr
import sewar as sewar_api

# numpy version
def SSIM_numpy(x_true, x_pred, data_range, sewar=False):
    r"""
    SSIM(Structural Similarity)锛岀粨鏋勭浉浼兼€э紝鏄竴绉嶈　閲忎袱骞呭浘鍍忕浉浼煎害鐨勬寚鏍囥€�
    缁撴瀯鐩镐技鎬х殑鑼冨洿涓�-1鍒�1銆傚綋涓ゅ紶鍥惧儚涓€妯′竴鏍锋椂锛孲SIM鐨勫€肩瓑浜�1銆�
    缁撴瀯鐩镐技搴︽寚鏁颁粠鍥惧儚缁勬垚鐨勮搴﹀皢缁撴瀯淇℃伅瀹氫箟涓虹嫭绔嬩簬浜害銆佸姣斿害鐨勶紝鍙嶆槧鍦烘櫙涓墿浣撶粨鏋勭殑灞炴€э紝
    骞跺皢澶辩湡寤烘ā涓轰寒搴︺€佸姣斿害鍜岀粨鏋勪笁涓笉鍚屽洜绱犵殑缁勫悎銆�
    鐢ㄥ潎鍊间綔涓轰寒搴︾殑浼拌锛屾爣鍑嗗樊浣滀负瀵规瘮搴︾殑浼拌锛屽崗鏂瑰樊浣滀负缁撴瀯鐩镐技绋嬪害鐨勫害閲忋€�
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SSIM value
    """
    if sewar:
        return sewar_api.ssim(x_true, x_pred, MAX=data_range)[0]

    return structural_similarity(x_true, x_pred, data_range=data_range, multichannel=True)


def MPSNR_numpy(x_true, x_pred, data_range):
    r"""
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
    Returns:
        float: Mean PSNR value
    """

    tmp = []
    for c in range(x_true.shape[-1]):
        tmp.append(peak_signal_noise_ratio(x_true[:, :, c], x_pred[:, :, c], data_range=data_range))
    return np.mean(tmp)


def SAM_numpy(x_true, x_pred, sewar=False):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details
    SAM鐢ㄦ潵璁＄畻涓や釜鏁扮粍涔嬮棿鐨勭浉浼兼€э紝鍏惰绠楃粨鏋滃彲鐪嬩綔涓ゆ暟缁勪箣闂翠綑寮﹁
    杈撳嚭缁撴灉鍊艰秺灏忚〃绀轰袱涓暟缁勮秺鍖归厤锛岀浉浼煎害瓒婇珮銆傚弽涔嬶紝琛ㄧず涓ゆ暟缁勮窛绂昏秺澶э紝鐩镐技搴﹁秺灏忋€�
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SAM value
    """
    if sewar:
        return sewar_api.sam(x_true, x_pred)

    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    dot_sum = np.sum(x_true * x_pred, axis=2)
    norm_true = norm(x_true, axis=2)
    norm_pred = norm(x_pred, axis=2)
    # runningwarning
    np.seterr(divide='ignore', invalid='ignore')
    cos_value = dot_sum / norm_pred / norm_true
    eps = 1e-6
    if 1.0 < cos_value.any() < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value.any() < -1.0:
        cos_value  = -1.0

    res = np.arccos(cos_value)
    is_nan = np.nonzero(np.isnan(res))
    # 杩斿洖鐨勬槸x涓殑涓嶄负0鐨勫厓绱犲潗鏍�
    # isnan杩斿洖鐨勬槸鏁扮粍瀵瑰簲鐨勭浉鍚屽ぇ灏忕殑甯冨皵鍨嬫暟缁�
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    sam = np.mean(res)
    return sam * 180 / np.pi


def SCC_numpy(ms, ps, sewar=False):
    r"""
    Look at paper:
    `A wavelet transform method to merge Landsat TM and SPOT panchromatic data` for details

    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SCC value
    """
    if sewar:
        return sewar_api.scc(ms, ps)

    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    scc = 0.0
    for i in range(ms.shape[2]):
        a = (ps_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        scc += pearsonr(a, b)[0]
    return scc / ms.shape[2]


def CC_numpy(ms, ps):
    r"""
    鐩稿叧绯绘暟CC
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: CC value
    """

    cc = 0.0
    for i in range(ms.shape[2]):
        a = (ps[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        cc += pearsonr(a, b)[0]
    return cc / ms.shape[2]


def Q4_numpy(ms, ps):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: Q4 value
    """

    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0, :] = 1
        return a * sign

    def product(a, b):
        a = a.reshape(a.shape[0], 1)
        b = b.reshape(b.shape[0], 1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r

    # ps = np.expand_dims(ps, axis=2)  # (H,W,C)
    # ms = np.expand_dims(ms, axis=2)
    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1] * imps.shape[0], imps.shape[2])  # (W*H, C)
    vec_ps = vec_ps.transpose(1, 0)  # (C, W*H)
    vec_ms = imms.reshape(imms.shape[1] * imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1, 0)  # (C, W*H)
    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1, 0) - m1).transpose(1, 0)  # (C, W*H)
    s1 = np.mean(np.sum(d1 * d1, axis=0))
    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)  # (C, W*H)
    s2 = np.mean(np.sum(d2 * d2, axis=0))
    Sc = np.zeros(vec_ms.shape)  # (C, W*H)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = product(d1[:, i], d2[:, i])
    C = np.mean(Sc, axis=1)
    Q4 = 4 * np.sqrt(np.sum(m1 * m1) * np.sum(m2 * m2) * np.sum(C * C)) / (s1 + s2) / (
                np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4

def Q8_numpy(ms, ps):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: Q8 value
    """

    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0, :] = 1
        return a * sign

    def product(a, b):
        a = a.reshape(a.shape[0], 1)
        b = b.reshape(b.shape[0], 1)
        R = np.dot(a, b.transpose())
        r = np.zeros(8)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3] - R[4, 4] - R[5, 5] - R[6, 6] - R[7, 7]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] + R[3, 2] + R[4, 5] + R[5, 4] + R[6, 7] - R[7, 6]
        r[2] = R[0, 2] - R[1, 3] + R[2, 4] + R[3, 1] + R[4, 6] + R[5, 7] + R[6, 0] + R[7, 5]
        r[3] = R[0, 3] + R[1, 2] - R[2, 5] + R[3, 0] + R[4, 7] + R[5, 6] + R[6, 1] + R[7, 4]

        r[4] = R[0, 4] + R[1, 5] + R[2, 6] - R[3, 7] + R[4, 0] + R[5, 1] + R[6, 2] + R[7, 3]
        r[5] = R[0, 5] + R[1, 4] + R[2, 7] + R[3, 6] - R[4, 1] + R[5, 0] + R[6, 3] + R[7, 2]
        r[6] = R[0, 6] + R[1, 7] + R[2, 0] + R[3, 5] + R[4, 2] - R[5, 3] + R[6, 4] + R[7, 1]
        r[7] = R[0, 7] + R[1, 6] + R[2, 1] + R[3, 4] + R[4, 3] + R[5, 2] - R[6, 5] + R[7, 0]
        return r

    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1] * imps.shape[0], imps.shape[2])  # (W*H, C)
    vec_ps = vec_ps.transpose(1, 0)  # (C, W*H)
    vec_ms = imms.reshape(imms.shape[1] * imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1, 0)  # (C, W*H)
    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1, 0) - m1).transpose(1, 0)  # (C, W*H)
    s1 = np.mean(np.sum(d1 * d1, axis=0))
    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)  # (C, W*H)
    s2 = np.mean(np.sum(d2 * d2, axis=0))
    Sc = np.zeros(vec_ms.shape)  # (C, W*H)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = product(d1[:, i], d2[:, i])
    C = np.mean(Sc, axis=1)
    Q8 = 4 * np.sqrt(np.sum(m1 * m1) * np.sum(m2 * m2) * np.sum(C * C)) / (s1 + s2) / (
                np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q8

def RMSE_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: RMSE value
    """
    if sewar:
        return sewar_api.rmse(ms, ps)

    d = (ms - ps) ** 2
    rmse = np.sqrt(np.sum(d) / (d.shape[0] * d.shape[1]))
    return rmse

def RASE(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    C1 = np.sum(np.power(ms[:, :, 0] - f[:, :, 0], 2)) / h / w
    C2 = np.sum(np.power(ms[:, :, 1] - f[:, :, 1], 2)) / h / w
    C3 = np.sum(np.power(ms[:, :, 2] - f[:, :, 2], 2)) / h / w
    C4 = np.sum(np.power(ms[:, :, 3] - f[:, :, 3], 2)) / h / w
    rase = np.sqrt((C1+C2+C3+C4)/4) * 100 / np.mean(ms)
    return rase

def RASE8(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    C1 = np.sum(np.power(ms[:, :, 0] - f[:, :, 0], 2)) / h / w
    C2 = np.sum(np.power(ms[:, :, 1] - f[:, :, 1], 2)) / h / w
    C3 = np.sum(np.power(ms[:, :, 2] - f[:, :, 2], 2)) / h / w
    C4 = np.sum(np.power(ms[:, :, 3] - f[:, :, 3], 2)) / h / w
    C5 = np.sum(np.power(ms[:, :, 4] - f[:, :, 4], 2)) / h / w
    C6 = np.sum(np.power(ms[:, :, 5] - f[:, :, 5], 2)) / h / w
    C7 = np.sum(np.power(ms[:, :, 6] - f[:, :, 6], 2)) / h / w
    C8 = np.sum(np.power(ms[:, :, 7] - f[:, :, 7], 2)) / h / w
    rase = np.sqrt((C1+C2+C3+C4+C5+C6+C7+C8)/8) * 100 / np.mean(ms)
    return rase


def QAVE(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    ms_mean = np.mean(ms,axis=-1)
    f_mean = np.mean(f,axis=-1)
    M1 = ms[:,:,0] - ms_mean
    M2 = ms[:,:,1] - ms_mean
    M3 = ms[:,:,2] - ms_mean
    M4 = ms[:,:,3] - ms_mean
    F1 = f[:, :, 0] - f_mean
    F2 = f[:, :, 1] - f_mean
    F3 = f[:, :, 2] - f_mean
    F4 = f[:, :, 3] - f_mean
    Qx = (1/4 - 1) * (np.power(M1,2) + np.power(M2,2) + np.power(M3,2) + np.power(M4,2))
    Qy = (1/4 - 1) * (np.power(F1,2) + np.power(F2,2) + np.power(F3,2) + np.power(F4,2))
    Qxy = (1/4 - 1) * (M1 * F1 + M2 * F2 + M3 * F3 + M4 * F4)
    Q = (4 * Qxy * ms_mean * f_mean) / ( (Qx + Qy) * ( np.power(ms_mean,2) + np.power(f_mean,2) ) + 2.2204e-16)
    qave = np.sum(Q) / h / w
    return qave

def QAVE8(I_ms,I_f):
    # f, ms = I_f, I_ms
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    ms_mean = np.mean(ms,axis=-1)
    f_mean = np.mean(f,axis=-1)
    M1 = ms[:,:,0] - ms_mean
    M2 = ms[:,:,1] - ms_mean
    M3 = ms[:,:,2] - ms_mean
    M4 = ms[:,:,3] - ms_mean
    M5 = ms[:, :, 4] - ms_mean
    M6 = ms[:, :, 5] - ms_mean
    M7 = ms[:, :, 6] - ms_mean
    M8 = ms[:, :, 7] - ms_mean
    F1 = f[:, :, 0] - f_mean
    F2 = f[:, :, 1] - f_mean
    F3 = f[:, :, 2] - f_mean
    F4 = f[:, :, 3] - f_mean
    F5 = f[:, :, 4] - f_mean
    F6 = f[:, :, 5] - f_mean
    F7 = f[:, :, 6] - f_mean
    F8 = f[:, :, 7] - f_mean
    Qx = (1/c - 1) * (np.power(M1,2) + np.power(M2,2) + np.power(M3,2) + np.power(M4,2) + np.power(M5,2) + np.power(M6,2) + np.power(M7,2) + np.power(M8,2))
    Qy = (1/c - 1) * (np.power(F1,2) + np.power(F2,2) + np.power(F3,2) + np.power(F4,2) + np.power(F5,2) + np.power(F6,2) + np.power(F7,2) + np.power(F8,2))
    Qxy = (1/c - 1) * (M1 * F1 + M2 * F2 + M3 * F3 + M4 * F4 + M5 * F5 + M6 * F6 + M7 * F7 + M8 * F8)
    Q = (c * Qxy * ms_mean * f_mean) / ((Qx + Qy) * (np.power(ms_mean,2) + np.power(f_mean,2)) + 2.2204e-16)
    qave = np.sum(Q) / h / w
    return qave

def ERGAS_numpy(ms, ps, ratio=0.25, sewar=False):
    r"""
    Look at paper:
    `Quality of high resolution synthesised images: Is there a simple criterion?` for details
    鐩稿鍏ㄥ眬鏃犵翰閲忚宸�
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: ERGAS value
    """
    if sewar:
        return sewar_api.ergas(ms, ps)

    m, n, d = ms.shape
    summed = 0.0
    for i in range(d):
        summed += (RMSE_numpy(ms[:, :, i], ps[:, :, i])) ** 2 / np.mean(ps[:, :, i]) ** 2
    ergas = 100 * ratio * np.sqrt(summed / d)
    return ergas


def UIQC_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: UIQC value
    """
    if sewar:
        return sewar_api.uqi(ms, ps)

    l = ms.shape[2]
    uiqc = 0.0
    for i in range(l):
        uiqc += QIndex_numpy(ms[:, :, i], ps[:, :, i])
    return uiqc / l


def QIndex_numpy(a, b):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (np.ndarray): one-channel image, shape like [H, W]
        b (np.ndarray): one-channel image, shape like [H, W]
    Returns:
        float: Q index value
    """
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2 + 1e-21) / (m1 ** 2 + m2 ** 2 + 1e-21)

    return Q


def D_lambda_numpy(l_ms, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_lambda value
    """
    if sewar:
        return sewar_api.d_lambda(l_ms, ps)

    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(QIndex_numpy(ps[:, :, i], ps[:, :, j]) - QIndex_numpy(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)

def D_lambda_k_numpy(l_ms, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_lambda value
    """
    if sewar:
        return sewar_api.d_lambda(l_ms, ps)

    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(Q4_numpy(ps[:, :, i], ps[:, :, j]) - Q4_numpy(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)

def D_s_numpy(l_ms, pan, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_s value
    """
    if sewar:
        return sewar_api.d_s(pan, l_ms, ps, r=3)

    L = ps.shape[2]
    # cv2.pyrDown() 浠庝竴涓珮鍒嗚鲸鐜囧ぇ灏哄鐨勫浘鍍忓悜涓婃瀯寤轰竴涓噾瀛楀锛堝昂瀵稿彉灏忥紝鍒嗚鲸鐜囬檷浣庯級
    l_pan = cv2.pyrDown(pan)  #, dstsize=(4, 3), borderType=None)
    l_pan = cv2.pyrDown(l_pan)
    # l_pan = cv2.resize(pan, dsize=None, fx=1/3, fy=1/3, interpolation=cv2.INTER_LINEAR)
    sum = 0.0
    for i in range(L):
        sum += np.abs(QIndex_numpy(ps[:, :, i], pan) - QIndex_numpy(l_ms[:, :, i], l_pan))
    return sum / L


def FCC_numpy(pan, ps):
    r"""
    Look at paper:
    `A wavelet transform method to merge landsat TM and SPOT panchromatic data` for details

    Args:
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: FCC value
    """
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    fcc = []
    for i in range(ps.shape[2]):
        a = convolve(ps[:, :, i], k, mode='constant').reshape(-1)
        b = convolve(pan, k, mode='constant').reshape(-1)
        fcc.append(pearsonr(b, a)[0])  # 璁＄畻涓や釜鏁扮粍鐨勭浉鍏崇郴鏁� 杈撳嚭鐨勭涓€涓€间负鐩稿叧绯绘暟锛涚浜屼釜鍊间负p鍊硷紝璇ュ€艰秺灏忚〃鏄庣浉鍏崇郴鏁拌秺鏄捐憲
    return np.max(fcc)


def SF_numpy(ps):
    r"""
    Look at paper:
    `Review of pixel-level image fusion` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SF value
    """
    f_row = np.mean((ps[:, 1:] - ps[:, :-1]) * (ps[:, 1:] - ps[:, :-1]))
    f_col = np.mean((ps[1:, :] - ps[:-1, :]) * (ps[1:, :] - ps[:-1, :]))
    return np.sqrt(f_row + f_col)


def SD_numpy(ps):
    r"""
    Look at paper:
    `A novel metric approach evaluation for the spatial enhancement of pansharpened images` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SD value
    """
    SD = 0.0
    for i in range(ps.shape[2]):
        SD += np.std(ps[:, :, i].reshape(-1))
    return SD / ps.shape[2]


# torch version
def SAM_torch(x_true, x_pred):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details

    Args:
        x_true (torch.Tensor): target images, shape like [N, C, H, W]
        x_pred (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SAM value of n images
    """
    dot_sum = torch.sum(x_true * x_pred, dim=1)
    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)
    a = torch.Tensor([1]).to(x_true.device, dtype=x_true.dtype)
    b = torch.Tensor([-1]).to(x_true.device, dtype=x_true.dtype)
    res = dot_sum / norm_pred / norm_true
    res = torch.max(torch.min(res, a), b)
    res = torch.acos(res) * 180 / 3.1415926
    sam = torch.mean(res)
    return sam


def sobel_torch(im):
    r"""
    Args:
        im (torch.Tensor): images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: images after sobel filter
    """
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.Tensor(sobel_kernel).to(im.device, dtype=im.dtype)
    return F.conv2d(im, weight)


def SCC_torch(x, y):
    r"""
    Args:
        x (torch.Tensor): target images, shape like [N, C, H, W]
        y (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SCC value of n images
    """
    x = sobel_torch(x)
    y = sobel_torch(y)

    vx = x - torch.mean(x, dim=(2, 3), keepdim=True)
    vy = y - torch.mean(y, dim=(2, 3), keepdim=True)
    scc = torch.sum(vx * vy, dim=(2, 3)) / torch.sqrt(torch.sum(vx * vx, dim=(2, 3))) / torch.sqrt(
        torch.sum(vy * vy, dim=(2, 3)))
    return torch.mean(scc)


def QIndex_torch(a, b, eps=1e-8):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (torch.Tensor): one-channel images, shape like [N, H, W]
        b (torch.Tensor): one-channel images, shape like [N, H, W]
    Returns:
        torch.Tensor: Q index value of all images
    """
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return torch.mean(4 * cov_ab * E_a * E_b / ( (var_a + var_b) * (E_a ** 2 + E_b ** 2) + eps) )


def D_lambda_torch(l_ms, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum / L / (L - 1)

def D_s_torch(l_ms, pan, l_pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L

# from dltool, something wrong
import math
# def q2n(gt, x, q_blocks_size, q_shift):
#     '''
#     '''
#     if isinstance(gt, torch.Tensor):
#         gt = gt.cpu().numpy().transpose(0, 2, 3, 1)
#         x = x.cpu().numpy().transpose(0, 2, 3, 1)
#
#     N, N1, N2, N3 = gt.shape  # 255 255 8
#     size2 = q_blocks_size  # 32
#
#     stepx = math.ceil(N1 / q_shift)  # 8
#     stepy = math.ceil(N2 / q_shift)  # 8
#
#     if stepy <= 0:
#         stepy = 1
#         stepx = 1
#
#     est1 = (stepx - 1) * q_shift + q_blocks_size - N1  # 1
#     est2 = (stepy - 1) * q_shift + q_blocks_size - N2  # 1
#     # if np.sum(np.array([est1 != 0, est2 != 0])) > 0:
#     # refref = np.zeros(shape=[N1+1, N2+1])
#     # fusfus = refref.copy()
#
#     for i in range(N3):
#         a1 = gt[..., 0]
#
#         ia1 = np.zeros(shape=[N, N1 + est1, N2 + est2])
#         ia1[:, : N1, : N2] = a1
#         ia1[:, :, N2:N2 + est2] = ia1[:, :, N2 - 1:-1:N2 - est2 + 1]
#         ia1[:, N1:N1 + est1, ...] = ia1[:, N1 - 1:-1:N1 - est1 + 1, ...]
#         if i == 0:
#             refref = ia1[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
#         else:
#             refref = np.concatenate([refref, ia1[..., np.newaxis]], axis=-1)
#         if i < N3:
#             gt = gt[..., 1:]
#
#     gt = refref
#
#     for i in range(N3):
#
#         a2 = x[..., 0]
#         ia2 = np.zeros(shape=[N, N1 + est1, N2 + est2])
#         ia2[:, : N1, : N2] = a2
#         ia2[:, :, N2:N2 + est2] = ia2[:, :, N2 - 1:-1:N2 - est2 + 1]
#         ia2[:, N1:N1 + est1, ...] = ia2[:, N1 - 1:-1:N1 - est1 + 1, ...]
#         if i == 0:
#             fusfus = ia2[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
#         else:
#             fusfus = np.concatenate([fusfus, ia2[..., np.newaxis]], axis=-1)
#
#         if i < N3:
#             x = x[..., 1:]
#     x = fusfus
#
#     x = np.array(x, dtype=np.uint16)
#     gt = np.array(gt, dtype=np.uint16)
#
#     _, N1, N2, N3 = gt.shape
#
#     if math.ceil(math.log2(N3)) - math.log2(N3) != 0:
#         Ndif = pow(2, math.ceil(math.log2(N3))) - N3
#         dif = np.zeros(shape=[N, N1, N2, Ndif], dtype=np.uint16)
#         gt = np.concatenate([gt, dif], axis=-1)
#         x = np.concatenate([x, dif], axis=-1)
#
#     _, _, _, N3 = gt.shape
#
#     valori = np.zeros(shape=[N, stepx, stepy, N3])
#
#     for j in range(stepx):
#         for i in range(stepy):
#             o = onions_quality(gt[:, j * q_shift:j * q_shift + q_blocks_size,
#                                i * q_shift: i * q_shift + size2, :],
#                                x[:, j * q_shift:j * q_shift + q_blocks_size,
#                                i * q_shift: i * q_shift + size2, :],
#                                q_blocks_size)
#             valori[:, j, i, :] = o
#     q2n_idx_map = np.sqrt(np.sum(valori ** 2, axis=-1))
#     q2n_index = np.mean(q2n_idx_map)
#     return q2n_index
#
# def onions_quality(dat1, dat2, size1):
#     dat1 = np.float64(dat1)
#     dat2 = np.float64(dat2)
#
#     dat2 = np.concatenate([dat2[..., 0, np.newaxis], -dat2[..., 1:]], axis=-1)
#     N, _, _, N3 = dat1.shape
#     size2 = size1
#
#     for i in range(N3):
#         a1, s, t = norm_blocco(np.squeeze(dat1[..., i]))
#         # print(s,t)
#         dat1[..., i] = a1
#         if s == 0:
#             if i == 0:
#                 dat2[..., i] = dat2[..., i] - s + 1
#             else:
#                 dat2[..., i] = -(-dat2[..., i] - s + 1)
#         else:
#             if i == 0:
#                 dat2[..., i] = ((dat2[..., i] - s) / t) + 1
#             else:
#                 dat2[..., i] = -(((-dat2[..., i] - s) / t) + 1)
#     m1 = np.zeros(shape=[N, N3])
#     m2 = m1.copy()
#
#     mod_q1m = 0
#     mod_q2m = 0
#     mod_q1 = np.zeros(shape=[size1, size2])
#     mod_q2 = np.zeros(shape=[size1, size2])
#
#     for i in range(N3):
#         m1[..., i] = np.mean(np.squeeze(dat1[..., i]))
#         m2[..., i] = np.mean(np.squeeze(dat2[..., i]))
#         mod_q1m += m1[..., i] ** 2
#         mod_q2m += m2[..., i] ** 2
#         mod_q1 += np.squeeze(dat1[..., i]) ** 2
#         mod_q2 += np.squeeze(dat2[..., i]) ** 2
#
#     mod_q1m = np.sqrt(mod_q1m)
#     mod_q2m = np.sqrt(mod_q2m)
#     mod_q1 = np.sqrt(mod_q1)
#     mod_q2 = np.sqrt(mod_q2)
#
#     termine2 = mod_q1m * mod_q2m  # 7.97
#     termine4 = mod_q1m ** 2 + mod_q2m ** 2  #
#     int1 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q1 ** 2)
#     int2 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q2 ** 2)
#     termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2 - 1)) * (mod_q1m ** 2 + mod_q2m ** 2)  # 17.8988  ** 2
#     mean_bias = 2 * termine2 / termine4  # 1
#     if termine3 == 0:
#         q = np.zeros(shape=[N, 1, N3])
#         q[:, :, N3 - 1] = mean_bias
#     else:
#         cbm = 2 / termine3
#         # 32 32 8
#         qu = onion_mult2D(dat1, dat2)
#         qm = onion_mult(m1.reshape(-1), m2.reshape(-1))
#         qv = np.zeros(shape=[N, N3])
#         for i in range(N3):
#             qv[..., i] = (size1 * size2) / ((size1 * size2) - 1) * np.mean(np.squeeze(qu[:, :, i]))
#         q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
#         q = q * mean_bias * cbm
#     return q
#
# def onion_mult2D(onion1, onion2):
#     _, _, _, N3 = onion1.shape
#
#     if N3 > 1:
#         L = N3 // 2
#         a = onion1[..., : L]
#         b = onion1[..., L:]
#         b = np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1)
#         c = onion2[..., : L]
#         d = onion2[..., L:]
#         d = np.concatenate([d[..., 0, np.newaxis], -d[..., 1:]], axis=-1)
#
#         if N3 == 2:
#             ris = np.concatenate([a * c - d * b, a * d + c * b], axis=-1)
#         else:
#             ris1 = onion_mult2D(a, c)
#             ris2 = onion_mult2D(d, np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1))
#             ris3 = onion_mult2D(np.concatenate([a[..., 0, np.newaxis], -a[..., 1:]], axis=-1), d)
#             ris4 = onion_mult2D(c, b)
#
#             aux1 = ris1 - ris2
#             aux2 = ris3 + ris4
#
#             ris = np.concatenate([aux1, aux2], axis=-1)
#     else:
#         ris = onion1 * onion2
#     return ris
#
#
# def onion_mult(onion1, onion2):
#     # _, N = onion1.shape
#     N = len(onion1)
#     if N > 1:
#
#         L = N // 2
#         a = onion1[:L]
#         b = onion1[L:]
#         # b[1:] = -b[1:]
#         b = np.append(np.array(b[0]), -b[1:])
#         c = onion2[:L]
#         d = onion2[L:]
#         # d[1:] = -d[1:]
#         d = np.append(np.array(d[0]), -d[1:])
#
#         if N == 2:
#             ris = np.append(a * c - d * b, a * d + c * b)
#         else:
#
#             ris1 = onion_mult(a, c)
#             # b[1:] = -b[1:]
#             ris2 = onion_mult(d, np.append(np.array(b[0]), -b[1:]))
#             # a[1:] = -a[1:]
#             ris3 = onion_mult(np.append(np.array(a[0]), -a[1:]), d)
#             ris4 = onion_mult(c, b)
#
#             aux1 = ris1 - ris2
#             aux2 = ris3 + ris4
#             ris = np.append(aux1, aux2)
#     else:
#         ris = np.array(onion1).reshape(-1) * np.array(onion2).reshape(-1)
#     return ris
import numpy as np
import math
import cv2
# from imresize_matlab import imresize

def norm_blocco(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c

def onion_mult(onion1, onion2):
    onion1 = onion1.copy()
    onion2 = onion2.copy()
    N = len(onion1)
    if N > 1:
        L = N // 2
        a = onion1[0:L]
        b = onion1[L:]
        b[1:] = -b[1:]
        c = onion2[0:L]
        d = onion2[L:]
        d[1:] = -d[1:]
        if N == 2:
            ris = np.array([a * c - d * b, a * d + c * b])
            return ris
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.append(b[0], -b[1:]))
            ris3 = onion_mult(np.append(a[0], -a[1:]), d)
            ris4 = onion_mult(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.append(aux1, aux2)
            return ris
    else:
        ris = onion1 * onion2
        return ris


def onion_mult2D(onion1, onion2):
    onion1 = onion1.copy()
    onion2 = onion2.copy()
    N3 = onion1.shape[2]
    if N3 > 1:
        L = N3 // 2
        a = onion1[:, :, 0:L]
        b = onion1[:, :, L:]
        b = np.append(b[:, :, 0, None], -b[:, :, 1:], axis=2)
        c = onion2[:, :, 0:L]
        d = onion2[:, :, L:]
        d = np.append(d[:, :, 0, None], -d[:, :, 1:], axis=2)

        if N3 == 2:
            ris = np.append(a * c - d * b, a * d + c * b, axis=2)
            return ris
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.append(b[:, :, 0, None], -b[:, :, 1:], axis=2))
            ris3 = onion_mult2D(np.append(a[:, :, 0, None], -a[:, :, 1:], axis=2), d)
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.append(aux1, aux2, axis=2)
            return ris
    else:
        ris = onion1 * onion2
        return ris


def onions_quality(dat1, dat2, size1):
    dat1 = dat1.copy()
    dat2 = dat2.copy()
    dat2 = np.append(dat2[:, :, 0, None], -dat2[:, :, 1:], axis=2)
    C = dat2.shape[2]
    for i in range(C):
        a1, s, t = norm_blocco(dat1[:, :, i])
        dat1[:, :, i] = a1
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -(-dat2[:, :, i] - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -(((-dat2[:, :, i] - s) / t) + 1)

    mod_q1 = np.zeros((size1, size1))
    mod_q2 = np.zeros((size1, size1))

    m1 = np.mean(dat1, axis=(0, 1))
    m2 = np.mean(dat2, axis=(0, 1))
    mod_q1m = np.sum(m1 ** 2)
    mod_q2m = np.sum(m2 ** 2)
    mod_q1 = mod_q1 + np.sum(dat1 ** 2, axis=2)
    mod_q2 = mod_q2 + np.sum(dat2 ** 2, axis=2)

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = (mod_q1m * mod_q2m)
    termine4 = (mod_q1m ** 2) + (mod_q2m ** 2)
    int1 = (size1 * size1) / ((size1 * size1) - 1) * np.mean(mod_q1 ** 2)
    int2 = (size1 * size1) / ((size1 * size1) - 1) * np.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size1) / ((size1 * size1) - 1) * ((mod_q1m ** 2) + (mod_q2m ** 2))
    mean_bias = 2 * termine2 / termine4
    if termine3 == 0:
        q = np.zeros((1, 1, C))
        q[:, :, C-1] = mean_bias
    else:
        cbm = 2 / termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = (size1 * size1) / ((size1 * size1) - 1) * np.mean(qu, axis=(0, 1))
        q = qv - (size1 * size1) / ((size1 * size1) - 1) * qm
        q = q * mean_bias * cbm
    return q


def Q2n(GT, Fused, Q_blocks_size=32, Q_shift=32):
    """
    get the Q4 or Q8 metric value
    the image shape is [C, H, W]  # [H, W, C]
    :param GT: ground truth image
    :param Fused: fused image
    :param Q_blocks_size: Block size of the Q-index locally applied
    :param Q_shift: Block shift of the Q-index locally applied
    :return: Q4 or Q8 value
    """
    GT = np.float64(GT)
    Fused = np.float64(Fused)
    H, W, C = GT.shape
    # GT = np.transpose(GT, (1, 2, 0))
    # Fused = np.transpose(Fused, (1, 2, 0))
    stepx = math.ceil(H / Q_shift)
    stepy = math.ceil(W / Q_shift)
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - H
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - W
    if est1 != 0 or est2 != 0:
        refref = []
        fusfus = []

        for i in range(C):
            a1 = GT[:, :, i]
            ia1 = np.zeros((H + est1, W + est2))
            ia1[0: H, 0: W] = a1
            ia1[:, W:W + est2] = ia1[:, W - 1:W - est2 - 1:-1]
            ia1[H:H + est1, :] = ia1[H - 1:H - est1 - 1:-1, :]
            np.append(refref, ia1, axis=2)
        GT = refref
        for i in range(C):
            a2 = Fused[:, :, i]
            ia2 = np.zeros(H + est1, W + est2)
            ia2[0: H, 0: W] = a2
            ia2[:, W:W + est2] = ia2[:, W - 1:W - est2 - 1:-1]
            ia2[H:H + est1, :] = ia2[H - 1:H - est1 - 1:-1, :]
            np.append(fusfus, ia2, axis=2)
        Fused = fusfus

    valori = np.zeros((stepx, stepy, C))

    for j in range(stepx):
        for i in range(stepy):
            valori[j, i, :] = onions_quality(GT[j * Q_shift:j * Q_shift + Q_blocks_size, i * Q_shift:i * Q_shift + Q_blocks_size, :],
                                             Fused[j * Q_shift:j * Q_shift + Q_blocks_size, i * Q_shift:i * Q_shift + Q_blocks_size, :],
                                             Q_blocks_size)
    Q2n_index_map = np.sqrt(np.sum((valori ** 2), axis=2))
    Q2n_index = np.mean(Q2n_index_map)
    return Q2n_index

# for d_lambda_k
def fir_filter_wind(Hd, w):
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    # h=h/np.sum(h)

    return h


def gaussian2d(N, std):
    t = np.arange(-(N - 1) / 2, (N + 1) / 2)
    # t=np.arange(-(N-1)/2,(N+2)/2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) / 2, (N + 1) / 2) / np.double(N - 1)
    # t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0

    return w


def genMTF(ratio, sensor, nbands):
    N = 41

    if sensor == 'QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22], dtype='float32')  # Band Order: B,G,R,NIR
    elif sensor == 'IKONOS':
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28], dtype='float32')  # Band Order: B,G,R,NIR
    elif sensor == 'GeoEye1' or sensor == 'WV4':
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23], dtype='float32')  # Band Order: B,G,R,NIR
    elif sensor == 'WV2':
        GNyq = [0.35 * np.ones(nbands), 0.27]
    elif sensor == 'WV3':
        GNyq = np.asarray([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], dtype='float32')
    else:
        GNyq = 0.3 * np.ones(nbands)

    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1 / ratio

    h = np.zeros((N, N, nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii])))
        H = gaussian2d(N, alpha)
        Hd = H / np.max(H)
        w = kaiser2d(N, 0.5)
        h[:, :, ii] = np.real(fir_filter_wind(Hd, w))

    return h

from scipy import ndimage
def MTF(I_MS, sensor, ratio):
    h = genMTF(ratio, sensor, I_MS.shape[2])

    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:, :, ii] = ndimage.filters.correlate(I_MS[:, :, ii], h[:, :, ii], mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)

# from FS_index.my_q2n import q2n
def D_lambda_K(fused, ms, ratio, sensor, S=32):
    if (fused.shape[0] != (ms.shape[0]) or fused.shape[1] != ms.shape[1]) == 1:
        print("The two images must have the same dimensions")
        return -1

    # N = fused.shape[0]
    # M = fused.shape[1]
    # if np.remainder(N,S-1) != 0:
    #     print("Number of rows must be multiple of the block size")
    #     return -1
    # if np.remainder(M,S-1) != 0:
    #     print("Number of columns must be multiple of the block size")
    #     return -1

    fused_degraded = MTF(fused, sensor, ratio)

    # fused_degraded = fused_degraded[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio),:]

    Q2n_index = Q2n(ms, fused_degraded, S, S)

    Dl = 1 - Q2n_index

    return Dl