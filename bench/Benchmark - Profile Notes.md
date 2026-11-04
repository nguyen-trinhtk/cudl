# GEMMs
Matrix size will be $M \times N \times K = 4096 \times 4096 \times 4096$
### Benchmark summary

TODO: draw a table
#### Naive kernel
In this variant, each thread compute one entry ($1 \times 1$) of $C$, with `x` dimensions representing rows and `y` representing cols. 
#### Global memory coalescing kernel
In this variant, each thread compute one entry ($1 \times 1$) of $C$. By swapping the role of `x` and `y`, neighboring threads are able to access contiguous data of $A$ and $C$, while sharing a column of $B$ (via broadcasting).
#### Shared memory caching kernel
In this variant, each thread compute one entry ($1 \times 1$) of $C$, while loading a $32 \times 32$ tile of $A$ and $B$ into shared memory.
#### 1D tiling kernel
In this variant, each thread compute a $8 \times 1$ block of $C$. Tile size is specified to be 64x8. 
#### 2D tiling kernel
In this variant, each thread compute a $8 \times 8$ block of $C$. Tile size is specified to be $64 \times 8$, so essentially a block covers a $64 \times 64$ block in $C$.
#### Vectorized load on 2D tiling kernel
The kernel is similar to the previous 2D tiling kernel, but with faster data access (dealing with a vector of $4$ floats at once).
