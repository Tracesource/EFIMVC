function S = cosine_similarity(A)
    epsilon = 1e-8;
    % 计算每行的L2范数
    norm_rows = sqrt(sum(A.^2, 2))+epsilon;
    % 处理零范数的情况，避免除以零（可选）
    norm_rows(norm_rows == 0) = 1; % 可选步骤，根据需求启用
    % 计算点积矩阵
    dot_products = A * A';
    % 计算范数乘积矩阵
    norm_matrix = norm_rows * norm_rows';
    % 计算余弦相似度矩阵
    S = dot_products ./ norm_matrix;
end