function rgb_img = HSI2RGB(file_path, spectral_excel)
% HSI2RGB - 高光谱图像转RGB彩色图
%
% 语法:
%   rgb_img = HSI2RGB(file_path)
%   rgb_img = HSI2RGB(file_path, spectral_excel)
%
% 输入参数:
%   file_path     - 高光谱图像文件夹路径
%   spectral_excel - 光谱响应数据Excel文件，默认'HSI2RGB20240517.xlsx'
%
% 输出参数:
%   rgb_img - RGB彩色图

if nargin < 2
    spectral_excel = 'HSI2RGB20240517.xlsx';
end

Connection = xlsread(spectral_excel);
X_m = Connection(:,2);
Y_m = Connection(:,3);
Z_m = Connection(:,4);

img_list = dir(strcat(file_path,'*.jpeg'));
img_num = length(img_list);

img = imread(strcat(file_path,img_list(1).name));
row = size(img,1);
col = size(img,2);

Xf = zeros(row,col);
Yf = zeros(row,col);
Zf = zeros(row,col);

if img_num > 0
    for j = 1:img_num
        image_name = img_list(j).name;
        I = double(imread(strcat(file_path,image_name)));
        Xf = I * X_m(j) + Xf;
        Yf = I * Y_m(j) + Yf;
        Zf = I * Z_m(j) + Zf;
    end
end

Xf = Xf / sum(X_m);
Yf = Yf / sum(Y_m);
Zf = Zf / sum(Z_m);

xyz_img = cat(3, Xf, Yf, Zf);
rgb_img = xyz2rgb(xyz_img);

r = uint8(rgb_img(:,:,1) * 22);
g = uint8(rgb_img(:,:,2) * 22.4);
b = uint8(rgb_img(:,:,3) * 22.6);

rgb_img = cat(3, r, g, b);
end
