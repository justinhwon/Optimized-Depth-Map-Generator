/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"
#include "utils.h"

/* Implements the displacement function */
float displacement_naive1(int dx, int dy) {
    return sqrt(dx * dx + dy * dy);
}

/* Helper function to return the square euclidean distance between two values. */
float square_euclidean_distanceN(float a, float b) {
    int diff = a - b;
    return diff * diff;
}

/* Helper function to return the square euclidean distance between two values. */
float square_euclidean_distance1(__m128 a, __m128 b, float sq_array[]) {
    __m128 diffs = _mm_sub_ps(a, b);
    __m128 squares = _mm_mul_ps(diffs, diffs);

    // doesn't make faster
    //__m128 sum =_mm_hadd_ps(squares, squares);
    //sum =_mm_hadd_ps(sum, sum);
    // get sum of all four in first 32 bits, only 1 array access

    _mm_storeu_ps((__m128 *) sq_array, squares);

    return sq_array[0] + sq_array[1] + sq_array[2] + sq_array[3];
    //return sq_array[0];
}

float square_euclidean_distance_tail(__m128 a, __m128 b, float sq_array[], int nums_left) {
    __m128 diffs = _mm_sub_ps(a, b);
    __m128 squares = _mm_mul_ps(diffs, diffs);

    // doesn't make faster
    //__m128 sum =_mm_hadd_ps(squares, squares);
    //sum =_mm_hadd_ps(sum, sum);
    // get sum of all four in first 32 bits, only 1 array access

    _mm_storeu_ps((__m128 *) sq_array, squares);

    float euclid = 0.0;
    for (int x = 0; x < nums_left; x++){
        euclid += sq_array[x];
    }
    return euclid;
}

void calc_depth_optimized(float *depth, float *left, float *right,
        int image_width, int image_height, int feature_width,
        int feature_height, int maximum_displacement) {
    // Array to be used whenever needed
    float squared_diff_array[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Naive implementation
    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            if (y < feature_height || y >= image_height - feature_height
                    || x < feature_width || x >= image_width - feature_width) {
                depth[y * image_width + x] = 0;
                continue;
            }
            float min_diff = -1;
            int min_dy = 0;
            int min_dx = 0;
            for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
                for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
                    if (y + dy - feature_height < 0
                            || y + dy + feature_height >= image_height
                            || x + dx - feature_width < 0
                            || x + dx + feature_width >= image_width) {
                        continue;
                    }
                    float squared_diff = 0;
                    for (int box_y = -feature_height; box_y <= feature_height; box_y++) {
                        // need to initialize box_x
                        int box_x;

                        // left_y and right_y don't change when looping over box_x
                        int left_y = y + box_y;
                        int right_y = y + dy + box_y;

                        for (box_x = -feature_width; box_x <= feature_width - 8; box_x+=8) {
                            int left_x0 = x + box_x;
                            int right_x0 = x + dx + box_x;
                            int left_x1 = x + 4 + box_x;
                            int right_x1 = x + 4 + dx + box_x; 

                            
                            float* left_ptr0 = left + (left_y * image_width + left_x0);
                            float* right_ptr0 = right + (right_y * image_width + right_x0);
                            float* left_ptr1 = left + (left_y * image_width + left_x1);
                            float* right_ptr1 = right + (right_y * image_width + right_x1);


                            __m128 leftVec0 = _mm_loadu_ps((__m128 *) left_ptr0);
                            __m128 rightVec0 = _mm_loadu_ps((__m128 *) right_ptr0);
                            __m128 leftVec1 = _mm_loadu_ps((__m128 *) left_ptr1);
                            __m128 rightVec1 = _mm_loadu_ps((__m128 *) right_ptr1);

                            //squared_diff += square_euclidean_distance1(leftVec, rightVec, squared_diff_array);

                            //inline fxn call to square_euclidean_distance1
                            __m128 diffs0 = _mm_sub_ps(leftVec0, rightVec0);
                            __m128 squares0 = _mm_mul_ps(diffs0, diffs0);
                            _mm_storeu_ps((__m128 *) squared_diff_array, squares0);
                            __m128 diffs1 = _mm_sub_ps(leftVec1, rightVec1);
                            __m128 squares1 = _mm_mul_ps(diffs1, diffs1);
                            _mm_storeu_ps((__m128 *) squared_diff_array+4, squares1);
                            squared_diff
                                += squared_diff_array[0] + squared_diff_array[1] + squared_diff_array[2] + squared_diff_array[3]
                                + squared_diff_array[4] + squared_diff_array[5] + squared_diff_array[6] + squared_diff_array[7];
                            
                        }
                        
                        for (box_x = box_x + 1; box_x <= feature_width - 4; box_x+=4) {
                            int left_x = x + box_x;
                            int right_x = x + dx + box_x;

                            
                            float* left_ptr = left + (left_y * image_width + left_x);
                            float* right_ptr = right + (right_y * image_width + right_x);


                            __m128 leftVec = _mm_loadu_ps((__m128 *) left_ptr);
                            __m128 rightVec = _mm_loadu_ps((__m128 *) right_ptr);

                            //squared_diff += square_euclidean_distance1(leftVec, rightVec, squared_diff_array);

                            //inline fxn call to square_euclidean_distance1
                            __m128 diffs = _mm_sub_ps(leftVec, rightVec);
                            __m128 squares = _mm_mul_ps(diffs, diffs);
                            _mm_storeu_ps((__m128 *) squared_diff_array, squares);
                            squared_diff += squared_diff_array[0] + squared_diff_array[1] + squared_diff_array[2] + squared_diff_array[3];
                            
                        }

                        // ignore later values in vector for tail case instead of looping with naive case
                        int numbers_left = feature_width - box_x + 1;

                        // if 1 don't loop (doesnt help just makes slower)
                        /*
                        if(numbers_left == 0){
                            continue;
                        }
                        if(numbers_left == 1){
                            int left_x = x + box_x;
                            int right_x = x + dx + box_x;
                            int diff = left[left_y * image_width + left_x] - right[right_y * image_width + right_x];
                            squared_diff += diff * diff;
                            continue;
                        }
                        */

                        int left_x = x + box_x;
                        int right_x = x + dx + box_x;

                        float* left_ptr = left + (left_y * image_width + left_x);
                        float* right_ptr = right + (right_y * image_width + right_x);


                        __m128 leftVec = _mm_loadu_ps((__m128 *) left_ptr);
                        __m128 rightVec = _mm_loadu_ps((__m128 *) right_ptr);

                        //squared_diff += square_euclidean_distance_tail(leftVec, rightVec, squared_diff_array, numbers_left);

                        //inline fxn call to square_euclidean_distance_tail
                        __m128 diffs = _mm_sub_ps(leftVec, rightVec);
                        __m128 squares = _mm_mul_ps(diffs, diffs);
                        _mm_storeu_ps((__m128 *) squared_diff_array, squares);
                        float euclid = 0.0;
                        for (int x = 0; x < numbers_left; x++){
                            euclid += squared_diff_array[x];
                        }
                        squared_diff += euclid;
                        
                        
                    }
                    if (min_diff == -1 || min_diff > squared_diff
                            || (min_diff == squared_diff
                                && sqrt(dx * dx + dy * dy) < sqrt(min_dx * min_dx + min_dy * min_dy))) { //inline fxn call
                        min_diff = squared_diff;
                        min_dx = dx;
                        min_dy = dy;
                    }
                }
            }
            if (min_diff != -1) {
                if (maximum_displacement == 0) {
                    depth[y * image_width + x] = 0;
                } else {
                    depth[y * image_width + x] = sqrt(min_dx * min_dx + min_dy * min_dy); //inline fxn call
                }
            } else {
                depth[y * image_width + x] = 0;
            }
        }
    }
}