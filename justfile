default:
    just --list

build:
    just build-diffpng

build-diffpng:
    g++ diffpng/diffpng.cpp diffpng/lodepng.cpp diffpng/lodepng.h -o build/diffpng
