#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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

void grayscale_filter_mpi(unsigned char* local_data, int local_height,
                          int width, int channels);
void gaussian_blur_filter_mpi(unsigned char* local_data, unsigned char* halo_top,
                              unsigned char* halo_bottom, int local_height,
                              int width, int channels, int rank, int size);
void sobel_edge_filter_mpi(unsigned char* local_data, unsigned char* halo_top,
                           unsigned char* halo_bottom, int local_height,
                           int width, int channels, int rank, int size);
void brightness_filter_mpi(unsigned char* local_data, int local_height,
                           int width, int channels, int brightness);

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input.ppm> <output.ppm> <filter>\n", argv[0]);
            fprintf(stderr, "Filters: grayscale, blur, edge, brighten\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    const char* filter_type = argv[3];
    
    Image* full_image = NULL;
    int width, height, channels;
    
    double start_time, end_time;
    
    // Root process loads image
    if (rank == 0) {
        printf("\n========================================\n");
        printf("Image Processing with MPI\n");
        printf("========================================\n");
        printf("Input:  %s\n", input_file);
        printf("Output: %s\n", output_file);
        printf("Filter: %s\n", filter_type);
        printf("MPI Processes: %d\n", size);
        printf("========================================\n\n");
        
        printf("Loading image...\n");
        full_image = load_image(input_file);
        if (!full_image) {
            fprintf(stderr, "Error: Could not load image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Image loaded: %dx%d, %d channels\n", 
               full_image->width, full_image->height, full_image->channels);
        
        width = full_image->width;
        height = full_image->height;
        channels = full_image->channels;
    }
    
    // Broadcast image dimensions
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate rows per process
    int rows_per_process = height / size;
    int remainder = height % size;
    
    // Adjust for remainder rows
    int local_height = rows_per_process;
    if (rank < remainder) {
        local_height++;
    }
    
    // Calculate displacement for each process
    int* sendcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_process;
            if (i < remainder) rows++;
            
            sendcounts[i] = rows * width * channels;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // Allocate local data
    int local_size = local_height * width * channels;
    unsigned char* local_data = (unsigned char*)malloc(local_size);
    
    // Scatter image data
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Scatterv(
        (rank == 0) ? full_image->data : NULL,
        sendcounts,
        displs,
        MPI_UNSIGNED_CHAR,
        local_data,
        local_size,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
    
    // Apply filter based on type
    if (strcmp(filter_type, "grayscale") == 0) {
        if (rank == 0) printf("Applying grayscale filter...\n");
        grayscale_filter_mpi(local_data, local_height, width, channels);
        
    } else if (strcmp(filter_type, "blur") == 0) {
        if (rank == 0) printf("Applying Gaussian blur filter...\n");
        
        // Allocate halo rows for neighbors
        int row_size = width * channels;
        unsigned char* halo_top = (unsigned char*)calloc(row_size, 1);
        unsigned char* halo_bottom = (unsigned char*)calloc(row_size, 1);
        
        // Exchange halo rows with neighbors
        MPI_Request requests[4];
        int req_count = 0;
        
        // Send/receive with upper neighbor
        if (rank > 0) {
            MPI_Isend(local_data, row_size, MPI_UNSIGNED_CHAR, 
                     rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(halo_top, row_size, MPI_UNSIGNED_CHAR, 
                     rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Send/receive with lower neighbor
        if (rank < size - 1) {
            MPI_Isend(local_data + (local_height - 1) * row_size, row_size, 
                     MPI_UNSIGNED_CHAR, rank + 1, 1, MPI_COMM_WORLD, 
                     &requests[req_count++]);
            MPI_Irecv(halo_bottom, row_size, MPI_UNSIGNED_CHAR, 
                     rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        
        // Apply blur with halo data
        gaussian_blur_filter_mpi(local_data, halo_top, halo_bottom, 
                                local_height, width, channels, rank, size);
        
        free(halo_top);
        free(halo_bottom);

    } else if (strcmp(filter_type, "edge") == 0) {
        if (rank == 0) printf("Applying Sobel edge detection filter...\n");

        // Allocate halo rows for neighbors
        int row_size = width * channels;
        unsigned char* halo_top = (unsigned char*)calloc(row_size, 1);
        unsigned char* halo_bottom = (unsigned char*)calloc(row_size, 1);

        // Exchange halo rows with neighbors
        MPI_Request requests[4];
        int req_count = 0;

        // Send/receive with upper neighbor
        if (rank > 0) {
            MPI_Isend(local_data, row_size, MPI_UNSIGNED_CHAR,
                     rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(halo_top, row_size, MPI_UNSIGNED_CHAR,
                     rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }

        // Send/receive with lower neighbor
        if (rank < size - 1) {
            MPI_Isend(local_data + (local_height - 1) * row_size, row_size,
                     MPI_UNSIGNED_CHAR, rank + 1, 1, MPI_COMM_WORLD,
                     &requests[req_count++]);
            MPI_Irecv(halo_bottom, row_size, MPI_UNSIGNED_CHAR,
                     rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }

        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Apply edge detection with halo data
        sobel_edge_filter_mpi(local_data, halo_top, halo_bottom,
                             local_height, width, channels, rank, size);

        free(halo_top);
        free(halo_bottom);

    } else if (strcmp(filter_type, "brighten") == 0) {
        if (rank == 0) printf("Applying brightness adjustment...\n");
        brightness_filter_mpi(local_data, local_height, width, channels, 50);

    } else {
        if (rank == 0) {
            fprintf(stderr, "Error: Unknown filter '%s'\n", filter_type);
        }
        free(local_data);
        if (rank == 0) free_image(full_image);
        MPI_Finalize();
        return 1;
    }
    
    // Gather results
    MPI_Gatherv(
        local_data,
        local_size,
        MPI_UNSIGNED_CHAR,
        (rank == 0) ? full_image->data : NULL,
        sendcounts,
        displs,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nProcessing time: %.6f seconds\n", end_time - start_time);
        
        printf("Saving output image...\n");
        save_image(output_file, full_image);
        printf("Done!\n\n");
        
        free(sendcounts);
        free(displs);
        free_image(full_image);
    }
    
    free(local_data);
    MPI_Finalize();
    return 0;
}

// ============================================
// FILTER IMPLEMENTATIONS
// ============================================

void grayscale_filter_mpi(unsigned char* local_data, int local_height, 
                          int width, int channels) {
    for (int i = 0; i < local_height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            
            unsigned char r = local_data[idx];
            unsigned char g = (channels > 1) ? local_data[idx + 1] : r;
            unsigned char b = (channels > 2) ? local_data[idx + 2] : r;
            
            unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            
            local_data[idx] = gray;
            if (channels > 1) local_data[idx + 1] = gray;
            if (channels > 2) local_data[idx + 2] = gray;
        }
    }
}

void gaussian_blur_filter_mpi(unsigned char* local_data, unsigned char* halo_top,
                              unsigned char* halo_bottom, int local_height, 
                              int width, int channels, int rank, int size) {
    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };
    
    // Need temporary buffer for output
    unsigned char* temp = (unsigned char*)malloc(local_height * width * channels);
    memcpy(temp, local_data, local_height * width * channels);
    
    // Process interior and border rows
    for (int i = 0; i < local_height; i++) {
        int start_j = (i == 0 || i == local_height - 1) ? 1 : 1;
        int end_j = width - 1;
        
        // Skip first and last row if at boundaries of entire image
        if ((rank == 0 && i == 0) || (rank == size - 1 && i == local_height - 1)) {
            continue;
        }
        
        for (int j = start_j; j < end_j; j++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        int ni = i + di;
                        int nj = j + dj;
                        
                        unsigned char pixel_val;
                        
                        // Get pixel from appropriate location
                        if (ni < 0) {
                            // Use halo_top
                            pixel_val = halo_top[nj * channels + c];
                        } else if (ni >= local_height) {
                            // Use halo_bottom
                            pixel_val = halo_bottom[nj * channels + c];
                        } else {
                            // Use local_data
                            pixel_val = temp[(ni * width + nj) * channels + c];
                        }
                        
                        sum += pixel_val * kernel[di + 1][dj + 1];
                    }
                }
                
                int out_idx = (i * width + j) * channels + c;
                local_data[out_idx] = (unsigned char)sum;
            }
        }
    }
    
    free(temp);
}

void sobel_edge_filter_mpi(unsigned char* local_data, unsigned char* halo_top,
                           unsigned char* halo_bottom, int local_height,
                           int width, int channels, int rank, int size) {
    // Sobel kernels for edge detection
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Need temporary buffer for output
    unsigned char* temp = (unsigned char*)malloc(local_height * width * channels);
    memcpy(temp, local_data, local_height * width * channels);

    // Process interior and border rows
    for (int i = 0; i < local_height; i++) {
        int start_j = 1;
        int end_j = width - 1;

        // Skip first and last row if at boundaries of entire image
        if ((rank == 0 && i == 0) || (rank == size - 1 && i == local_height - 1)) {
            continue;
        }

        for (int j = start_j; j < end_j; j++) {
            float sum_x = 0.0, sum_y = 0.0;

            // Calculate gradient using first channel (or grayscale)
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di;
                    int nj = j + dj;

                    unsigned char pixel_val;

                    // Get pixel from appropriate location
                    if (ni < 0) {
                        // Use halo_top
                        pixel_val = halo_top[nj * channels];
                    } else if (ni >= local_height) {
                        // Use halo_bottom
                        pixel_val = halo_bottom[nj * channels];
                    } else {
                        // Use local_data
                        pixel_val = temp[(ni * width + nj) * channels];
                    }

                    sum_x += pixel_val * Gx[di + 1][dj + 1];
                    sum_y += pixel_val * Gy[di + 1][dj + 1];
                }
            }

            // Calculate magnitude
            float magnitude = sqrt(sum_x * sum_x + sum_y * sum_y);
            if (magnitude > 255.0) magnitude = 255.0;

            unsigned char edge_value = (unsigned char)magnitude;

            // Set all channels to the edge value
            int out_idx = (i * width + j) * channels;
            for (int c = 0; c < channels; c++) {
                local_data[out_idx + c] = edge_value;
            }
        }
    }

    free(temp);
}

void brightness_filter_mpi(unsigned char* local_data, int local_height,
                           int width, int channels, int brightness) {
    int local_size = local_height * width * channels;
    
    for (int i = 0; i < local_size; i++) {
        int new_val = local_data[i] + brightness;
        if (new_val > 255) new_val = 255;
        if (new_val < 0) new_val = 0;
        local_data[i] = (unsigned char)new_val;
    }
}

// ============================================
// IMAGE I/O (PPM Format)
// ============================================

Image* load_image(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    char format[3];
    int width, height, max_val;
    
    fscanf(fp, "%2s\n%d %d\n%d\n", format, &width, &height, &max_val);
    
    if (strcmp(format, "P6") != 0) {
        fclose(fp);
        return NULL;
    }
    
    Image* img = create_image(width, height, 3);
    fread(img->data, 1, width * height * 3, fp);
    fclose(fp);
    
    return img;
}

void save_image(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
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