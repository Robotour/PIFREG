% cutandregistration_demo - 高光谱图像波段配准演示脚本
%
% 本脚本演示如何使用KEREN算法进行高光谱图像波段间配准

%% 确定路径
img_filename = '2024-06-26_16-12-50-white\';
raw_path = strcat('Tongue cut dataset\', img_filename);
cut_path = strcat('Test cut dataset\', img_filename);
registration_path = strcat('registration result2\', img_filename);

%% 分割舌体区域
raw_list = dir(strcat(raw_path, '*.jpeg'));
img = imread(strcat(raw_path, raw_list(30).name));
imshow(img)
p = getrect;
p = round(p);

%% 将数据保存为cell图像
img_num = length(raw_list);
img_cell = cell(img_num, 1);
img_cell2 = cell(img_num, 1);

if img_num > 0
    for j = 1:img_num
        img_name = raw_list(j).name;
        I = double(imread(strcat(raw_path, img_name)));
        img_cell{j} = imcrop(I, p);
    end
end

%% 主成分分析，得到第一主成分的区域
a = size(img_cell{1}, 1);
b = size(img_cell{1}, 2);
c = length(img_cell);
image_numpy = zeros(a, b, c);

for i = 1:c
    image_numpy(:, :, i) = img_cell{i};
end

% 主成分分析
RES = pca_function(image_numpy);
pc1 = RES(:, :, 1);

% 第一主成分二值化的区域
pc1_level = graythresh(pc1) * 0.5;
pc1_BW = imbinarize(pc1, pc1_level);

% 只保留第一主成分区域的信息
for i = 1:c
    temp = img_cell{i};
    non_zero_temp = temp(pc1_BW);
    mean_value = mean(non_zero_temp);
    temp = temp .* pc1_BW;
    temp(temp == 0) = mean_value;
    img_cell2{i} = temp;
end

%% 归一化
img_cell_non = cell(img_num, 1);
for i = 1:img_num
    temp = img_cell{i};
    max_data = max(max(temp));
    min_data = min(min(temp));
    img_cell_non{i} = (temp - min_data) / (max_data - min_data);
end

%% KEREN配准
[delta_est, phi_est] = keren2(img_cell_non);

%% 应用变换
if img_num > 0
    for i = 1:img_num
        img_name = raw_list(i).name;
        I = uint8(shiftandrotate(img_cell{i}, delta_est(i, 2), delta_est(i, 1), phi_est(i)));
        imwrite(I, strcat(registration_path, img_name));
    end
end

%% 对比伪彩色图
contrast_path = strcat('Test cut dataset\', img_filename);
rgb_img1 = HSI2RGB(contrast_path);
rgb_img2 = HSI2RGB(registration_path);

figure
subplot(121)
imshow(rgb_img1), title("配准前")
subplot(122)
imshow(rgb_img2), title("配准后")
