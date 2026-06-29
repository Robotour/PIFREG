function [delta_est, phi_est] = keren2(img_list)
% keren2 - 改进版Lucas-Kanade金字塔光流配准算法
%
% 相比keren.m，增加了金字塔层数和迭代次数
%
% 语法:
%   [delta_est, phi_est] = keren2(img_list)
%
% 输入参数:
%   img_list - 图像元胞数组，第一幅为参考图像
%
% 输出参数:
%   delta_est - 每幅图像的平移量 (N, 2)
%   phi_est   - 每幅图像的旋转角度 (N,)

img_tem{1} = img_list{1};
for img_num = 2:length(img_list)
    lp = fspecial('ga',3,1);
    img_pro{1} = img_list{img_num};
    pyrlevel_num = 5;
    for i=2:pyrlevel_num
        img_tem{i} = imresize(conv2(img_tem{i-1},lp,'same'),0.5,'bicubic');
        img_pro{i} = imresize(conv2(img_pro{i-1},lp,'same'),0.5,'bicubic');
    end
    
    stot = zeros(1,3);
    for pyrlevel=pyrlevel_num:-1:1
        f0 = img_tem{pyrlevel};
        f1 = img_pro{pyrlevel};
        
        [y0,x0]=size(f0);
        xmean=x0/2; ymean=y0/2;
        x=kron(-xmean:xmean-1,ones(y0,1));
        y=kron(ones(1,x0),(-ymean:ymean-1)');
     
        sigma=1;  
        g1 = zeros(y0,x0); g2 = g1; g3 = g1;
        for i=1:y0
            for j=1:x0
                g1(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))*(i-ymean)/2/pi/sigma^2;
                g2(i,j)=-exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))*(j-xmean)/2/pi/sigma^2;
                g3(i,j)= exp(-((i-ymean)^2+(j-xmean)^2)/(2*sigma^2))/2/pi/sigma^2;
            end
        end
        
        a=real(ifft2(fft2(f1).*fft2(g2)));  
        c=real(ifft2(fft2(f1).*fft2(g1))); 
        b=real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0).*fft2(g3))); 
        R=c.*x-a.*y;
        
        a11 = sum(sum(a.*a)); a12 = sum(sum(a.*c)); a13 = sum(sum(R.*a));
        a21 = sum(sum(a.*c)); a22 = sum(sum(c.*c)); a23 = sum(sum(R.*c)); 
        a31 = sum(sum(R.*a)); a32 = sum(sum(R.*c)); a33 = sum(sum(R.*R));
        b1 = sum(sum(a.*b)); b2 = sum(sum(c.*b)); b3 = sum(sum(R.*b));
        Ainv = [a11 a12 a13; a21 a22 a23; a31 a32 a33]^(-1);
        s = Ainv*[b1; b2; b3];
        st = s;
        
        it=1;
        while ((abs(s(1))+abs(s(2))>0.1)&&it<50)
            f0_ = shift(f0,-st(1),-st(2));
            f0_ = imrotate(f0_,-st(3)*180/pi,'bicubic','crop');
            b = real(ifft2(fft2(f1).*fft2(g3)))-real(ifft2(fft2(f0_).*fft2(g3)));
            s = Ainv*[sum(sum(a.*b)); sum(sum(c.*b)); sum(sum(R.*b))];
            st = st+s;
            it = it+1;
        end
        
        st(3)=-st(3)*180/pi;
        st = st';
        st(1:2) = st(2:-1:1);
        stot = [2*stot(1:2)+st(1:2) stot(3)+st(3)];
        if pyrlevel>1
            img_pro{pyrlevel-1} = imrotate(img_pro{pyrlevel-1},-stot(3),'bicubic','crop');
            img_pro{pyrlevel-1} = shift(img_pro{pyrlevel-1},2*stot(2),2*stot(1));
        end
    end
    delta_est(img_num,:) = stot(1:2);
    phi_est(img_num) = stot(3);
    img_tem{1}=(img_tem{1}+shiftandrotate(img_list{img_num},stot(2),stot(1),stot(3)))/2;
    imshow(uint8(img_tem{1}));
end
end
