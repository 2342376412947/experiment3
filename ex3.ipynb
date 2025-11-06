
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 将图像转换为灰度图
# 读取图像并检查是否成功
img_a = cv2.imread("c.png")
img_b = cv2.imread("d.png")

if img_a is None or img_b is None:
    raise ValueError("无法读取图像文件。请确保图片路径正确且文件存在。")

# 转换为灰度图
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

##初始化SIFT检测器并提取特征
sift = cv2.SIFT_create()  # 文档中此处为Cv2（大写C），修正为cv2（小写c）
kp_a, des_a = sift.detectAndCompute(gray_a, None)  # 图像a的关键点和描述符
kp_b, des_b = sift.detectAndCompute(gray_b, None)  # 图像b的关键点和描述符
img1_kp = cv2.drawKeypoints(
    image=img_a,
    keypoints=kp_a,
    outImage=None,
    color=(0, 255, 0),  # 特征点标记为绿色
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 显示尺度（大小）和方向（箭头）
)

# 绘制图像2的特征点：彩色标记，显示尺度和方向
img2_kp = cv2.drawKeypoints(
    image=img_b,
    keypoints=kp_b,
    outImage=None,
    color=(0, 0, 255),  # 特征点标记为红色
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 4. 显示或保存可视化结果
cv2.imshow("Image 1 with SIFT Keypoints", img1_kp)
cv2.imshow("Image 2 with SIFT Keypoints", img2_kp)



cv2.waitKey(0)
cv2.destroyAllWindows()
# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 文档中FLANN后多空格，已修正
search_params = dict(checks=50)  # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_a, des_b, k=2)  # k=2表示每个特征点返回2个最佳匹配
# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 比率阈值通常取0.7-0.8
        good_matches.append(m)
# 基本诊断输出
print(f"图像A关键点: {len(kp_a)}, 图像B关键点: {len(kp_b)}")
print(f"Lowe 筛选后匹配数: {len(good_matches)}")

# 初步绘制匹配（后面会用RANSAC内点重绘）
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像b的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像a的关键点

# 使用RANSAC算法估计单应矩阵(透视变换矩阵)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# mask为内点掩码，统计内点数量并在匹配图中只显示内点
if mask is None:
    print("findHomography 未返回掩码，可能匹配不足或失败。")
    inlier_count = 0
else:
    inlier_count = int(mask.sum())
print(f"RANSAC 内点数量: {inlier_count} / {len(good_matches)}")

# 如果内点过少，提示并提前退出
if inlier_count < 4:
    raise RuntimeError("内点太少，无法计算可靠的单应矩阵（需要至少4个内点）。")

# 使用内点重新绘制匹配图（只显示内点匹配）
inlier_matches = [gm for gm, m in zip(good_matches, mask.ravel()) if m]
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, inlier_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 额外诊断信息：打印单应矩阵和数值特性，检测是否存在异常缩放/畸变
print("单应矩阵 H:\n", H)
try:
    cond_H = np.linalg.cond(H)
except Exception:
    cond_H = float('inf')
print(f"H 条件数: {cond_H}")

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]  # 文档中为wb，修正为w_b

# 计算图像b变换后的四个角点坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)  # 文档中为w_b,e，修正为w_b,0
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸(包含所有像素)
all_corners = np.concatenate([
    dst_corners, 
    np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)  # 文档中缺失部分坐标值，已补充
], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 创建平移矩阵，确保所有像素都在可见区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)  # 文档中缺失0，已补充

# 对图像b进行透视变换和平移
fus_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,  # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
)

# 将图像a复制到拼接结果的对应位置
fus_img[-y_min:h_a - y_min, -x_min:w_a - x_min] = img_a  # 文档中为xmin，修正为x_min

# 显示匹配关键点和拼接结果
plt.figure(figsize=(20, 20))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
plt.title('图像A')
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.title('图像B')
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB))
plt.title('特征匹配')
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(fus_img, cv2.COLOR_BGR2RGB))
plt.title('拼接结果')
plt.show()  # 显示所有图像
print("按任意键关闭图像窗口...")
plt.waitforbuttonpress()  # 等待按键后再关闭窗口
cv2.imshow("Feature Matching Results", match_img)  # 显示匹配图
cv2.imwrite("feature_matches.jpg", match_img)      # 保存匹配图到本地
cv2.waitKey(0)
cv2.destroyAllWindows()