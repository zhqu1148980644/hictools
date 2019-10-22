typedef struct LineIterator {
    int max_dim;
    int max_coor[20];
    int coordinates[20];
    int strides[20];
    int back_strides[20];

} LineIterator;


typedef struct ArrayInfo {
    int ndim;
    int size;
    int itemsize;
    int shapes[20];
    int strides[20];

} ArrayInfo;

typedef enum ExtendMode {
    NEAREST,
    WRAP,
    REFLECT,
    MIRROR,
    CONSTANT

} ExtendMode;


typedef struct ConvInfo {
    int *shift;
    int size1;
    int size2;
    int length;
    int nlines;
    int step;
    int symmetric;
    int nonzero;
    int numpoints;
    ExtendMode mode;

} ConvInfo;

#define SKIP_ZERO1(curp, step) { \
    if ((*(curp)) == 0) {        \
        (curp) += step;          \
        continue;                \
    }                            \
}

#define SKIP_ZERO2(arrayp, index, output) { \
    if (*((arrayp) + index) == 0) {         \
        ++output;                           \
        continue;                           \
    }                                       \
}

#define GOTO_NEXTLINE(liter, datap) {                            \
    for (_i = (liter)->max_dim; _i >= 0; --_i){                  \
        if ((liter)->coordinates[_i] < (liter)->max_coor[_i]) {  \
            (liter)->coordinates[_i]++;                          \
            datap += (liter)->strides[_i];                       \
            break;                                               \
        }                                                        \
        else {                                                   \
            (liter)->coordinates[_i] = 0;                        \
            datap -= (liter)->back_strides[_i];                  \
        }                                                        \
    }                                                            \
}


#define FILL_CONSTANT() {                                        \
    curp = linep;                                                \
    for (item_i = 0; item_i < size1; ++item_i) {                 \
        if (nonzero) SKIP_ZERO1(curp, step)                      \
        sum = 0;                                                 \
        mid = -item_i;                                           \
        for (ki = -size1; ki < mid; ++ki)                        \
            sum += cval * k[ki];                                 \
        for (ki = mid; ki < size2 + 1; ++ki)                     \
            sum += curp[shift[ki]] * k[ki];                      \
        output[curp - arrayp] = sum;                             \
        curp += step;                                            \
    }                                                            \
    curp = linep + (length - size2) * step;                      \
    for (item_i = length - size2; item_i < length; ++item_i) {   \
        if (nonzero) SKIP_ZERO1(curp, step)                      \
        sum = 0;                                                 \
        mid = length - item_i;                                   \
        for (ki = -size1; ki < mid; ++ki)                        \
            sum += curp[shift[ki]] * k[ki];                      \
        for (ki = mid; ki < size2 + 1; ++ki)                     \
            sum += cval * k[ki];                                 \
        output[curp - arrayp] = sum;                             \
        curp += step;                                            \
    }                                                            \
}


#define SYMMETRY_PO() {                                           \
    curp = linep + line_step;                                     \
    for (item_i = size1; item_i < length - size2; ++item_i) {     \
        if (nonzero) SKIP_ZERO1(curp, step)                       \
        sum = (*curp) * (*k);                                     \
        for (ki = -size1; ki < 0; ++ki)                           \
            sum += (curp[shift[ki]] + curp[-shift[ki]]) * k[ki];  \
        output[curp - arrayp] = sum;                              \
        curp += step;                                             \
    }                                                             \
}


#define SYMMETRY_NE() {                                            \
    curp = linep + line_step;                                      \
    for (item_i = size1; item_i < length - size2; ++item_i) {      \
        if (nonzero) SKIP_ZERO1(curp, step)                        \
        sum = (*curp) * (*k);                                      \
        for (ki = -size1; ki < 0; ++ki)                            \
            sum += (curp[shift[ki]] - curp[-shift[ki]]) * k[ki];   \
        output[curp - arrayp] = sum;                               \
        curp += step;                                              \
    }                                                              \
}

#define NO_SYMMTRY() {                                             \
    curp = linep + line_step;                                      \
    for (item_i = size1; item_i < length - size2; ++item_i) {      \
        if (nonzero) SKIP_ZERO1(curp, step)                        \
        for (ki = -size1, sum = 0; ki < size2 + 1; ++ki)           \
            sum += curp[shift[ki]] * k[ki];                        \
        output[curp - arrayp] = sum;                               \
        curp += step;                                              \
    }                                                              \
}

template<typename T>
void correlate1d_normal(const T *arrayp, T *output, const T *k, T cval,
                        const ConvInfo *ci, LineIterator *liter) {
    int size1 = ci->size1, size2 = ci->size2;
    int length = ci->length, step = ci->step, nlines = ci->nlines;
    int *shift = ci->shift + size1;
    int line_step = size1 * step;
    int _i, line_i, item_i, ki, mid;
    int nonzero = ci->nonzero;
    const T *curp, *linep = arrayp;
    T sum;
    k += size1;

    switch (ci->mode) {
        case NEAREST:
            break;
        case WRAP:
            break;
        case REFLECT:
            break;
        case MIRROR:
            break;
        case CONSTANT:
            if (ci->symmetric > 0) {
                for (line_i = 0; line_i < nlines; ++line_i) {
                    SYMMETRY_PO()
                    FILL_CONSTANT()
                    GOTO_NEXTLINE(liter, linep)
                }
            } else if (ci->symmetric < 0) {
                for (line_i = 0; line_i < nlines; ++line_i) {
                    SYMMETRY_NE()
                    FILL_CONSTANT()
                    GOTO_NEXTLINE(liter, linep)
                }
            } else {
                for (line_i = 0; line_i < nlines; ++line_i) {
                    NO_SYMMTRY()
                    FILL_CONSTANT()
                    GOTO_NEXTLINE(liter, linep)
                }
            }
            break;
        default:
            break;
    }

}


template<typename T>
void correlate1d_points(const T *array, T *output, const int *indexes,
                      const T *k, const ConvInfo *ci) {
    int nonzero = ci->nonzero, numpoints = ci->numpoints;
    int i, ki, index, size1 = ci->size1, size2 = ci->size2;
    int *shift = ci->shift + size1;
    T sum;
    k += size1;

    if (ci->symmetric > 0) {
        for (i = 0; i < numpoints; ++i) {
            index = *indexes++;
            if (nonzero) SKIP_ZERO2(array, index, output)
            sum = array[index] * *k;
            for (ki = -size1; ki < 0; ++ki)
                sum += (array[index + shift[ki]] + array[index - shift[ki]]) * k[ki];
            *output++ = sum;
        }
    } else if (ci->symmetric < 0) {
        for (i = 0; i < numpoints; ++i) {
            index = *indexes++;
            if (nonzero) SKIP_ZERO2(array, index, output)
            sum = array[index] * *k;
            for (ki = -size1; ki < 0; ++ki)
                sum += (array[index + shift[ki]] - array[index - shift[ki]]) * k[ki];
            *output++ = sum;
        }
    } else {
        for (i = 0; i < numpoints; ++i) {
            index = *indexes++;
            if (nonzero) SKIP_ZERO2(array, index, output)
            for (ki = -size1, sum = 0; ki < size2 + 1; ++ki)
                sum += array[index + shift[ki]] * k[ki];
            *output++ = sum;
        }
    }
}

