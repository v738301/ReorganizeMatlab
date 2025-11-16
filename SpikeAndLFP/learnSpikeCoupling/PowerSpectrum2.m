data = load('04_ECoG-1.mat');
x = data.ECoG;
t = data.t;

dt = t(2) - t(1);
N = length(t);
T = N*dt;

x = hanning(N) .* x;
xf = fft(x - mean(x));
Sxx = 2 .* dt^2 ./ T .* (xf .* conj(xf));
Sxx = real(Sxx(1:round(length(x)/2)));

df = 1/T;
fNQ = 1/ dt /2;
faxis = 0:df:fNQ-df;

figure;
plot(faxis, Sxx)