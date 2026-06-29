function RES = pca_function(image_cube)
% pca_function - 主成分分析提取舌体区域
%
% 语法:
%   RES = pca_function(image_cube)
%
% 输入参数:
%   image_cube - 高光谱图像立方体 (H x W x N)，N为波段数
%
% 输出参数:
%   RES - 主成分分析结果 (H x W x 3)，返回前3个主成分

[a, b, c] = size(image_cube);
image_2d = reshape(image_cube, a * b, c);

% 标准化
X = zscore(image_2d);

% 计算相关系数矩阵
R = corrcoef(image_2d);

% 计算特征值和特征向量
[V, D] = eig(R);
lambda = diag(D);
lambda = lambda(end:-1:1);
V = rot90(V)';

% 计算主成分
m = c;
F = zeros(size(image_2d, 1), m);
for i = 1:m
    ai = V(:,i)';
    Ai = repmat(ai, size(image_2d, 1), 1);
    F(:, i) = sum(Ai .* X, 2);
end

% 重塑为图像格式
m = 3;
feature_after_PCA = F(:, 1:m);
RES = reshape(feature_after_PCA, a, b, m);

% 第二主成分取反
test = RES(:,:,2);
if test(1,1) > 0
    RES(:,:,2) = imcomplement(RES(:,:,2));
end

% 归一化到0-255
RES(RES < 0) = NaN;
for i = 1:m
    a_mat = RES(:,:,i);
    min_val = min(a_mat(:), [], 'omitnan');
    max_val = max(a_mat(:), [], 'omitnan');
    RES(:,:,i) = uint8((a_mat - min_val) / (max_val - min_val) * 255);
end
end
