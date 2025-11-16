data = load('ECoG-1.mat');
E1 = data.E1;
E2 = data.E2;
t = data.t;
dt = t(2) - t(1);
N = length(t);
T = N*dt;
df = 1/T;
scale = 2 * dt^2 /T;

%%
f = [0:N/2-1]*df;
xf = [];
yf = [];
phi8 = [];
phi24 = [];

for i = 1:size(E1,1)
    xff = fft(E1(i,:));
    xf = [xf; xff(1:N/2)];
    yff = fft(E2(i,:));
    yf = [yf; yff(1:N/2)];
    Sxy = scale .* (xff .* conj(yff));
    phi8(i) = angle(Sxy(find(f==8)));
    phi24(i) = angle(Sxy(find(f==24)));
end

%%

Sxx = mean(scale .* (xf .* conj(xf)));
Syy = mean(scale .* (yf .* conj(yf)));
Sxy = mean(scale .* (xf .* conj(yf)));

cohr = abs(Sxy) ./ (sqrt(Sxx) .* sqrt(Syy));

%%

figure; hold on;
plot(f, Sxx)

figure; hist(phi8)
figure; hist(phi24)