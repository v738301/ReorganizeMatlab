clear all
close all

%%
path = '/Users/hsiehkunlin/Downloads/Case-Studies-Python-master/matfiles';
data = load(fullfile(path,"03_EEG-1.mat"));

EEG = data.EEG;
t = data.t;
N = length(EEG);
dt = t(2) - t(1);
T = N * dt;
x = EEG;
%%

figure; hold on;
plot(t, EEG)

%%

figure; hold on;
plot(t(1:25), EEG(1:25))

%%
mn = mean(x);
vr = var(x);
sd = std(x);

%%
lags = -length(x)+1:1:length(x)-1;

ac = (1 / N) * xcorr(x - mean(x), x - mean(x));
inds = find(abs(lags) <= 100);


figure; hold on;
plot(lags(inds)*dt, ac(inds))

%%
L = 0;
inds = 1:100;
figure; hold on;
plot(t(inds), x(inds), 'k');
plot(t(inds), x(inds+L), 'r')

%%
L = round(1/2*1/60/dt);
inds = 1:100;
figure; hold on;
plot(t(inds), x(inds), 'k');
plot(t(inds), x(inds+L), 'r')

fprintf('%d Autocorrelation \n', ac(lags==L))

%%
xf = fft(x - mean(x));
Sxx = 2 .* dt .^ 2 / T .* (xf .* conj(xf));
Sxx = Sxx(1:round(length(x)./2));

df = 1/ T;
fNQ = 1 / dt / 2;
figure;
semilogx([1:length(Sxx)]*df, 10*log10(Sxx./max(Sxx)))
ylim([-60,0])

%%
tt = linspace(0,1,1000);
x = cos(2 * pi * 10 * tt);
figure; plot(tt, x, 'k')

fj = 10                                %# Set the frequency
fj_sin = sin(-2 * pi * fj * tt)        %# Construct the sine wave
fj_cos = cos(-2 * pi * fj * tt)        %# ... and cosine wave

figure; hold on;
plot(tt, x, 'k')         %# Plot the data
plot(tt, fj_sin, 'r--')  %# ... and the sine
plot(tt, fj_cos, 'ro') %# ... and cosine

%%
%% Generate a test signal
fs = 1000;  % sampling rate
t = 0:1/fs:2-1/fs;  % 2 seconds
x = sin(2*pi*10*t) + 0.5*sin(2*pi*25*t) + 0.2*randn(size(t));

N = length(x);

% Method 2: Power Spectrum via Wiener-Khinchin
% Step 1: Compute autocorrelation
[acf, lags] = xcorr(x-mean(x),x-mean(x));  % biased gives proper scaling
% acf = acf .* 1./(N-abs(lags));
acf = acf .* 1./N;

[Pxx_direct, f] = pwelch(x, [], [], [], fs);

xf = fft(x - mean(x));
Sxx = 2 .* 1/fs .^ 2 / 2 .* (xf .* conj(xf));
Sxx = Sxx(1:round(length(x)./2));

df = 1/ 2;
fNQ = 1 / 1/fs / 2;
figure;
plot([1:length(Sxx)]*df, 10*log10(Sxx./max(Sxx)))

% Or use the full autocorrelation directly
nfft = length(acf);
Pxx_WK = sqrt(fft(acf).*conj(fft(acf))) ./ N;  % FFT of autocorrelation

% Take one-sided spectrum
Pxx_WK = Pxx_WK(1:floor(nfft/2)+1);
f_WK = (0:floor(nfft/2)) * fs / nfft;

% Plot comparison
figure;
subplot(2,1,1)
plot(f, 10*log10(Pxx_direct), 'b', 'LineWidth', 2)
hold on
plot([1:length(Sxx)]*df, 10*log10(Sxx./max(Sxx)))
% plot(f_WK, 10*log10(Pxx_WK), 'r--', 'LineWidth', 1.5)
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
legend('Direct FFT', 'Via W-K Theorem')
title('Power Spectrum Comparison')
grid on

subplot(2,1,2)
plot(lags/fs, acf)
xlabel('Lag (s)')
ylabel('Autocorrelation')
title('Autocorrelation Function')
grid on

%%
clear all; close all; clc;

% --------------------------------------------------------------------
% Parameters
fs = 1000;
T  = 2;
dt = 1/fs;
t  = 0:dt:T-dt;
x  = sin(2*pi*10*t) + 0.5*sin(2*pi*25*t);
N  = length(x);

% --------------------------------------------------------------------
% 1) Direct FFT → zero-pad to 2N-1 for fair comparison
xf   = fft(x);
Sxx  = abs(xf).^2 / N;                    % divide by original N
f    = (0:N/2)' * fs/N;

% Single-sided
idx_ss = 1 : floor(N/2) + 1;
Sxx_ss = Sxx(idx_ss);
Sxx_ss(2:end-1) = 2 * Sxx_ss(2:end-1);
f_ss = f(idx_ss);

% --------------------------------------------------------------------
% 2) Autocorrelation (biased → already /N)
[acf, lags] = xcorr(x, 'biased');         % length 2N-1, already /N
acf = acf(:).';                           % row vector
acff = fft(acf);

Nac = length(acff);
idx_ac = 1 : floor(Nac/2) + 1;
Pxx_WK = real(acff);                      % should match Sxx
Pxx_WK_ss = Pxx_WK(idx_ac);
Pxx_WK_ss(2:end-1) = 2 * Pxx_WK_ss(2:end-1);
f_ss_wk = (0:Nac/2)' * fs/Nac;
% --------------------------------------------------------------------
% 4) Plot
figure('Color','w','Position',[100 100 900 500]);
plot(f_ss, Sxx_ss, 'b', 'LineWidth', 2); hold on;
plot(f_ss_wk, Pxx_WK_ss, 'r', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power (dB/Hz)');
title('Power Spectrum – Direct FFT vs. Wiener–Khinchin (T=200s)');
legend('Direct FFT (zero-padded)', 'FFT\{R_{xx}\} (W-K)', 'Location','best');
grid on; axis tight;

%%

% Generate a random signal
fs = 1000; % Sampling frequency
t = 0:1/fs:1-1/fs; % Time vector
x = cos(2*pi*50*t); % Example signal

% Calculate ACF
Rxx = xcorr(x, 'unbiased'); % Using unbiased estimator

% Calculate PSD from ACF
N = length(x);
% The autocorrelation length is 2*N - 1. Use N for the FFT length.
PSD_from_ACF = fftshift(fft(Rxx, N)); 
% Normalize the PSD
PSD_from_ACF = abs(PSD_from_ACF) / N; 

% Calculate FFT of the signal
X = fftshift(fft(x, N));
% Calculate PSD from FFT
PSD_from_FFT = abs(X).^2 / N; 

% Create frequency vector
f = linspace(-fs/2, fs/2 - fs/N, N);

% Plot both PSDs
figure;
plot(f, PSD_from_ACF, 'b', 'DisplayName', 'PSD from ACF');
hold on;
plot(f, PSD_from_FFT, 'r', 'DisplayName', 'PSD from FFT');
title('Wiener-Khinchin Theorem Demonstration');
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB)');
legend;
grid on;

