function B = lin2rgb(varargin)
%LIN2RGB Apply gamma correction to linear RGB values
%
%   B = LIN2RGB(A) applies a gamma correction to the linear RGB values in
%   matrix A so that B is in sRGB space, which is suitable for display. To
%   apply the gamma correction for Adobe RGB (1998), use the 'ColorSpace'
%   name-value pair.
%
%   B = LIN2RGB(___,Name,Value,...) specifies additional options as
%   name-value pairs:
%
%     'ColorSpace'  -  Color space of the output image:
%                      'srgb' (default) | 'adobe-rgb-1998'
%
%                      Default: 'srgb'
%
%     'OutputType'  -  Data type of the returned RGB values:
%                      'double' | 'single' | 'uint8' | 'uint16'
%
%                      Default: class(A)
%
%   Class Support
%   -------------
%   A must be a real, non-sparse array of one of the following classes:
%   uint8, uint16, single or double.
%
%   Notes
%   -----
%   [1] The gamma correction to transform linear RGB tristimulous values
%   into sRGB is defined by the following parametric curve:
%     f(u) = -f(-u),               u < 0
%     f(u) = c*u,             0 <= u < d
%     f(u) = a*u^gamma + b,        u >= d
%
%   where u represents a color value and with parameters:
%     a = 1.055
%     b = -0.055
%     c = 12.92
%     d = 0.0031308
%     gamma = 1/2.4
%
%   [2] The gamma correction to transform linear RGB tristimulous values
%   into Adobe RGB (1998) is done using a simple power function:
%
%     v = u^gamma,       if u >= 0
%     v = -(-u)^gamma,   otherwise
%
%   with gamma = 1/2.19921875.
%
%   References
%   ----------
%   [1] Ebner, Marc. Gamma Correction, Color Constancy.
%       John Wiley & Sons, 2007. ISBN 978-0-470-05829-9.
%   [2] Section 4.3.4.2., Adobe RGB (1998) Color Image Encoding,
%       May 2005, p.12.
%
%   Examples
%   --------
%   [1] Apply the gamma correction specified by the sRGB standard
%       to linear RGB values.
%
%     lin2rgb([0.5 0.5 0.5])
%
%   [2] Apply the gamma correction specified by the Adobe RGB (1998)
%       standard to linear RGB values.
%
%     lin2rgb([0.5 0.5 0.5], 'ColorSpace', 'adobe-rgb-1998')
%
%   [3] Apply the sRGB gamma correction to an image containing linear
%       RGB values for display on a typical computer monitor.
%
%     % Read in the image file. The image data it contains is the raw
%     % sensor data after correcting the black level and scaling to 16 bits
%     % per pixel. No other processing has been done.
%     A = imread('foosballraw.tiff');
%
%     % Demosaic the image to interpolate linear RGB colors.
%     A_demosaicked = demosaic(A,'rggb');
%
%     % Apply the sRGB gamma correction for display
%     % and encode in double precision.
%     A_sRGB = lin2rgb(A_demosaicked, 'OutputType', 'double');
%
%     % Display the gamma-corrected image.
%     figure
%     imshowpair(A_demosaicked, A_sRGB, 'montage')
%     title('Minimally processed sensor data without and with sRGB gamma correction')
%
%   See also RGB2LIN.

%   Copyright 2016 The MathWorks, Inc.

[A,colorSpace,outputType] = parseInputs(varargin{:});

% Convert to floating point for the conversion
if ~isa(A,'double')
    A = im2single(A);
end

if strcmp(colorSpace, 'srgb')
    B = linearRGBToSRGB(A);
else
    B = linearRGBToAdobeRGB(A);
end

% Convert to the desired output type
convert = str2func(['im2' outputType]);
B = convert(B);

%--------------------------------------------------------------------------
function y = linearRGBToSRGB(x)
% Curve parameters
gamma = cast(1/2.4,'like',x);
a     = cast(1.055,'like',x);
b     = cast(-0.055,'like',x);
c     = cast(12.92,'like',x);
d     = cast(0.0031308,'like',x);

y = zeros(size(x),'like',x);

in_sign = -2 * (x < 0) + 1;
x = abs(x);

lin_range = (x < d);
gamma_range = ~lin_range;

y(gamma_range) = a * exp(gamma .* log(x(gamma_range))) + b;
y(lin_range) = c * x(lin_range);

y = y .* in_sign;

%--------------------------------------------------------------------------
function y = linearRGBToAdobeRGB(x)
gamma = cast(1/2.19921875,'like',x);
y = ( exp(gamma .* log(abs(x))) ) .* sign(x);

%--------------------------------------------------------------------------
function [A,colorSpace,outputType] = parseInputs(varargin)

narginchk(1,5);

parser = inputParser();
parser.FunctionName = mfilename;

% A
validateImage = @(x) validateattributes(x, ...
    {'single','double','uint8','uint16'}, ...
    {'real','nonsparse','nonempty'}, ...
    mfilename,'A',1);
parser.addRequired('A', validateImage);

% NameValue 'ColorSpace': 'srgb' or 'adobe-rgb-1998'
defaultColorSpace = 'srgb';
validateChar = @(x) validateattributes(x, ...
    {'char','string'}, ...
    {'scalartext'}, ...
    mfilename, 'ColorSpace');
parser.addParameter('ColorSpace', ...
    defaultColorSpace, ...
    validateChar);

% NameValue 'OutputType': 'single', 'double', 'uint8', 'uint16'
defaultOutputType = -1;
parser.addParameter('OutputType', ...
    defaultOutputType, ...
    validateChar);

parser.parse(varargin{:});
inputs = parser.Results;
A = inputs.A;
colorSpace = inputs.ColorSpace;
outputType = inputs.OutputType;

if isequal(outputType, defaultOutputType)
    outputType = class(A);
end

% Additional validation
colorSpace = validatestring( ...
    colorSpace, ...
    {'srgb','adobe-rgb-1998'}, ...
    mfilename, 'ColorSpace');

outputType = validatestring( ...
    outputType, ...
    {'single','double','uint8','uint16'}, ...
    mfilename, 'OutputType');
