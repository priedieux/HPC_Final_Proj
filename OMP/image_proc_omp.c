#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Function prototypes
Image* load_image(const char* filename);
void save_image(const char* filename, Image* img);
void free_image(Image* img);
Image* create_image(int width, int height, int channels);

void grayscale_filter(Image* input, Image* output, int thread_count);
void gaussian_blur_filter(Image* input, Image* output, int thread_count);
void sobel_edge_filter(Image* input, Image* output, int thread_count);
void brightness_filter(Image* input, Image* output, int brightness, int thread_count);

void print_usage(const char* prog_name);

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    const char* filter_type = argv[3];
    int thread_count = atoi(argv[4]);
    
    if (thread_count < 1) {
        fprintf(stderr, "Error: thread_count must be positive\n");
        return 1;
    }
    
    printf("\n========================================\n");
    printf("Image Processing with OpenMP\n");
    printf("========================================\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Filter: %s\n", filter_type);
    printf("Threads: %d\n", thread_count);
    printf("========================================\n\n");
    
    // Load image
    printf("Loading image...\n");
    Image* input = load_image(input_file);
    if (!input) {
        fprintf(stderr, "Error: Could not load image %s\n", input_file);
        return 1;
    }
    printf("Image loaded: %dx%d, %d channels\n", 
           input->width, input->height, input->channels);
    
    // Create output image
    Image* output = create_image(input->width, input->height, input->channels);
    
    // Apply filter and measure time
    double start_time = 0.0;
    double end_time = 0.0;
    
#ifdef _OPENMP
    start_time = omp_get_wtime();
#endif
    
    if (strcmp(filter_type, "grayscale") == 0) {
        grayscale_filter(input, output, thread_count);
    } else if (strcmp(filter_type, "blur") == 0) {
        gaussian_blur_filter(input, output, thread_count);
    } else if (strcmp(filter_type, "edge") == 0) {
        sobel_edge_filter(input, output, thread_count);
    } else if (strcmp(filter_type, "brighten") == 0) {
        brightness_filter(input, output, 50, thread_count);
    } else {
        fprintf(stderr, "Error: Unknown filter '%s'\n", filter_type);
        fprintf(stderr, "Available filters: grayscale, blur, edge, brighten\n");
        free_image(input);
        free_image(output);
        return 1;
    }
    
#ifdef _OPENMP
    end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("\nProcessing time: %.6f seconds\n", elapsed);
#endif
    
    // Save output
    printf("Saving output image...\n");
    save_image(output_file, output);
    printf("Done!\n\n");
    
    free_image(input);
    free_image(output);
    
    return 0;
}

// ============================================
// FILTER IMPLEMENTATIONS
// ============================================

void grayscale_filter(Image* input, Image* output, int thread_count) {
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    
    printf("Applying grayscale filter...\n");
    
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            
            unsigned char r = input->data[idx];
            unsigned char g = (channels > 1) ? input->data[idx + 1] : r;
            unsigned char b = (channels > 2) ? input->data[idx + 2] : r;
            
            // Weighted average for human perception
            unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            
            output->data[idx] = gray;
            if (channels > 1) output->data[idx + 1] = gray;
            if (channels > 2) output->data[idx + 2] = gray;
            if (channels > 3) output->data[idx + 3] = input->data[idx + 3]; // Alpha
        }
    }
}

void gaussian_blur_filter(Image* input, Image* output, int thread_count) {
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    
    // 3x3 Gaussian kernel
    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };
    
    printf("Applying Gaussian blur filter...\n");
    
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                
                // Apply 3x3 kernel
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        int idx = ((i + di) * width + (j + dj)) * channels + c;
                        sum += input->data[idx] * kernel[di + 1][dj + 1];
                    }
                }
                
                int out_idx = (i * width + j) * channels + c;
                output->data[out_idx] = (unsigned char)sum;
            }
        }
    }
    
    // Copy borders
    for (int i = 0; i < height; i++) {
        if (i == 0 || i == height - 1) {
            for (int j = 0; j < width; j++) {
                int idx = (i * width + j) * channels;
                for (int c = 0; c < channels; c++) {
                    output->data[idx + c] = input->data[idx + c];
                }
            }
        } else {
            int idx_left = (i * width) * channels;
            int idx_right = (i * width + width - 1) * channels;
            for (int c = 0; c < channels; c++) {
                output->data[idx_left + c] = input->data[idx_left + c];
                output->data[idx_right + c] = input->data[idx_right + c];
            }
        }
    }
}

void sobel_edge_filter(Image* input, Image* output, int thread_count) {
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    
    // Sobel kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    printf("Applying Sobel edge detection filter...\n");
    
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum_x = 0, sum_y = 0;
            
            // Calculate gradient for first channel (or grayscale)
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int idx = ((i + di) * width + (j + dj)) * channels;
                    unsigned char pixel = input->data[idx];
                    
                    sum_x += pixel * Gx[di + 1][dj + 1];
                    sum_y += pixel * Gy[di + 1][dj + 1];
                }
            }
            
            // Magnitude
            float magnitude = sqrt(sum_x * sum_x + sum_y * sum_y);
            if (magnitude > 255) magnitude = 255;
            
            int out_idx = (i * width + j) * channels;
            unsigned char edge_value = (unsigned char)magnitude;
            
            for (int c = 0; c < channels; c++) {
                output->data[out_idx + c] = edge_value;
            }
        }
    }
}

void brightness_filter(Image* input, Image* output, int brightness, int thread_count) {
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    
    printf("Applying brightness adjustment (+%d)...\n", brightness);
    
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            
            for (int c = 0; c < channels; c++) {
                int new_val = input->data[idx + c] + brightness;
                if (new_val > 255) new_val = 255;
                if (new_val < 0) new_val = 0;
                output->data[idx + c] = (unsigned char)new_val;
            }
        }
    }
}

// ============================================
// IMAGE I/O (PPM Format - No external libs needed)
// ============================================

Image* load_image(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    char format[3];
    int width, height, max_val;
    /* Read PPM header. Expecting: P6\n<width> <height>\n<max_val>\n */
    int scanned = fscanf(fp, "%2s %d %d %d", format, &width, &height, &max_val);
    if (scanned != 4) {
        fclose(fp);
        return NULL;
    }

    if (strcmp(format, "P6") != 0) {
        fclose(fp);
        return NULL;
    }

    if (width <= 0 || height <= 0) {
        fclose(fp);
        return NULL;
    }

    /* Consume one byte (likely a single newline) before the binary data */
    int c = fgetc(fp);
    if (c == EOF) {
        fclose(fp);
        return NULL;
    }

    Image* img = create_image(width, height, 3);
    if (!img) {
        fclose(fp);
        return NULL;
    }

    size_t expected = (size_t)width * (size_t)height * 3;
    size_t nread = fread(img->data, 1, expected, fp);
    if (nread != expected) {
        free_image(img);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return img;
}

void save_image(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    if (fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height) < 0) {
        fclose(fp);
        return;
    }

    size_t expected = (size_t)img->width * (size_t)img->height * (size_t)img->channels;
    size_t nw = fwrite(img->data, 1, expected, fp);
    (void)nw; /* ignore in release; could check and report errors if needed */
    fclose(fp);
}

Image* create_image(int width, int height, int channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (unsigned char*)calloc(width * height * channels, sizeof(unsigned char));
    return img;
}

void free_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <input.ppm> <output.ppm> <filter> <num_threads>\n", prog_name);
    fprintf(stderr, "\nFilters:\n");
    fprintf(stderr, "  grayscale - Convert to grayscale\n");
    fprintf(stderr, "  blur      - Gaussian blur\n");
    fprintf(stderr, "  edge      - Sobel edge detection\n");
    fprintf(stderr, "  brighten  - Increase brightness\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s input.ppm output.ppm blur 4\n", prog_name);
}