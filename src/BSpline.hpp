#pragma once

#include "Tools/Tools.hpp"

#include <array>

using namespace Tools;

template< size_t DomDim, size_t AmbDim, size_t Degree, typename Scal_in, typename Scal_out, bool Closed = false>
class BSpline
{
    static_assert(DomDim >= 1);
    
    static_assert(AmbDim >= 1);
    
    static_assert(Degree >= 1);
    
public:
    
    using Int      = size_t;
    
    using Index_T  = std::array<Int,            DomDim>;
//    using Grid_T   = std::array<const Scal_in *,DomDim>;
    
    using KnotVector_T  = std::vector<Scal_in>;
    using Grid_T        = std::array<KnotVector_T,DomDim>;
    
    using Input_T  = std::array<Scal_in, DomDim>;
    using Output_T = std::array<Scal_out,AmbDim>;
    
    using CoefficientVector_T = std::array<Scal_in, Degree + 1>;
    using CoefficientMatrix_T = std::array<CoefficientVector_T,DomDim>;
    
    static constexpr Int corner_count = (1 << DomDim);
    
protected:
    
    Grid_T  grid;                           // a vector of grid points for each dimension
    Index_T dims;
    const   Scal_out * const      values;   // we do not copy the values; we just store a pointer
    
    mutable Index_T               grid_idx = {0};
//    mutable std::array<Input_T,2> t = {{{0},{1}}};
    mutable Input_T               x = {0};
    mutable Output_T              y = {0};
    
    mutable CoefficientMatrix_T B = {}; // Degree+1 B-spline coefficients for each dimension
    
    constexpr Index_T O = {0};
    
public:
    
//    BSpline( const Grid_T & grid_, const Index_T & dims_, const Scal_out * const values_ )
//    :   grid(grid_)
//    ,   dims(dims_)
//    ,   values(values_)
//    {}
    
    template< typename I>
    BSpline( const Scal_in * const * const grid_, const I * const dims_, const Scal_out * const values_ )
    :   values(values_)
    {
        std::copy( &dims_[0], &dims_[DomDim], &dims[0] );
        std::copy( &grid_[0], &grid_[DomDim], &grid[0] );
        
//        for( Int k = 0; k < DomDim; ++k )
//        {
//            // We have to add padding to each grid vector.
//            grid[k] = KnotVector_T( dims[k] + 2 * Degree );
//
//            std::copy( &grid_[0], &grid_[DomDim], &grid[p] );
//
//            for( Int j = 0; j < Degree; ++j )
//            {
//                grid[p][j] = grid[p][dims[k] + Degree - 1];
//            }
//        }
        
        
    }
    
    void Evaluate( const Scal_in * const x_, Scal_out * const y_ )
    {
        Read(x_);
        Find();
        Eval();
        Write(y_);
    }
    
    bool Increment( Index_T & idx, const Index_T & lo, const Index_T & hi )
    {
        // Move idx one step into the tensor grid box defined by lo and hi.
        // Last entry of idx is the "fastest" index.
        // Returns `true` if succeeded and `false` if we have `idx >= hi` after incrementing.
        for( Int k = DomDim; k --> 0; )
        {
            if( (++idx[k]) < hi[k] )
            {
                return true;
            }
            else
            {
                idx[k] = lo[k];
            }
        }
        
        return false;
    }
    
    
    const Index_T & GridIndex() const
    {
        return grid_idx;
    }

    Int GlobalIndex( const Index_T & idx ) const
    {
        if constexpr( DomDim == 1 )
        {
            return idx[0];
        }
        else
        {
            Int global_idx = 0;
            
            for( Int k = 0; k < DomDim - 1; ++k )
            {
                global_idx = (global_idx + idx[k]) * dims[k+1];
            }
            
            global_idx += idx[DomDim-1];

            return global_idx;
        }
    }
    
    void Read( const Scal_in * const x_ )
    {
        std::copy( &x_[0], &x_[DomDim], &x[0] );
    }
    
    void Write( Scal_out * const y_ ) const
    {
        std::copy( &y[0], &y[AmbDim], &y_[0] );
    }
    
    void Eval()
    {
        // TODO: Tensor product evaluation.
        
//        Scal_in volume = static_cast<Scal_in>(1);
//
//        for( Int k = 0; k < DomDim; ++k )
//        {
//            volume *= t[0][k] + t[1][k];
//        }
//
//        const Scal_in volume_inv = static_cast<Scal_in>(1) / volume;
//
//        std::fill( &y[0], &y[AmbDim], static_cast<Scal_out>(0) );
//
//        for( Int kdx = 0; kdx < corner_count; ++kdx )
//        {
//            // For each corner of the grid cuboid, we have to find the global position `global_idx` of the corner in the grid and its coeffcient `coeff`
//
//            Index_T idx = grid_index;
//
//            Scal_in coeff = volume_inv;
//
//            for( Int k = 0; k < DomDim; ++k )
//            {
//                Int bit  = (kdx >> (DomDim-1-k)) & 1;
//                idx[k]  += bit;
//                coeff   *= t[bit][k];
//            }
//
//            const Int global_idx = AmbDim * GlobalIndex(idx);
//
//            auto coeff_converted = static_cast<Scal_out>(coeff);
//
//            for( Int k = 0; k < AmbDim; ++k )
//            {
//                y[k] += coeff_converted * values[global_idx+k];
//            }
//        }
    }
    
    
    void Find()
    {
        for( Int k = 0; k < DomDim; ++k )
        {
            Find( x[k], k );
        }
    }
    
    void Find( const Scal_in z, const Int k )
    {
        const Scal_in * const g = grid[k];
    
        Int a = Degree;
        Int b = dims[k] + Degree - 1;
        
        Scal_in g_a = g[a];
        Scal_in g_b = g[b];
        
        if( z < g_a )
        {
            ComputeCoefficients( z, 0, k );
        }
        else if ( z > g_b )
        {
            ComputeCoefficients( z, b + Degree, k );
        }
        else
        {
            // Binary search
            while( b > a + 1 )
            {
                Int     c   = a + (b-a)/2;
                Scal_in g_c = g[c];
                
                if( z <= g_c )
                {
                    b   = c;
                    g_b = g_c;
                }
                else
                {
                    a   = c;
                    g_a = g_c;
                }
            }
        
            ComputeCoefficients( z, a, k );
        }
    }

    
    void ComputeCoefficients( const Scal_in z, const Int i, const Int k )
    {
        // Cox-de Boor algorithm
        
        grid_idx[k] = i;

        const KnotVector_T & t = grid[k];
        
        CoefficientVector_T & b = B[k];
        
        b[Degree] = 1.;
        
        auto alpha = [&t]( Int j, Int q, Scal_in z)
        {
            return (t[j] < t[j + q]) ? (z - t[j]) / (t[j + q] - t[j]) : static_cast<Scal_in>(0);
        }
        
        // TODO: Check this!
        for( Int q = 1, q <= Degree; ++q )  // Beware: Unconventional loop here.
        {
            Scal_in factor = alpha(i-q,q,z);
            
            b[Degree - q] = (static_cast<Scal_in>(1) - factor) * b[Degree-q+1];

            for( Int j = Degree - q + 1, j < Degree - 1; ++j )
            {
                b[j] *= factor;
                
                factor = alpha(i-Degree+j,q,z);
                
                b[j] += (static_cast<Scal_in>(1) - factor) * b[j+1];
            }
            
            b[Degree] *= factor;
        }
    }
    
}; // class BSpline
