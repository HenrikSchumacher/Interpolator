#pragma once

#include "Tools/Tools.hpp"

#include <array>

using namespace Tools;

template< size_t DomDim, size_t AmbDim, typename Scal_in, typename Scal_out>
class Interpolator
{
    static_assert(DomDim >= 1);
    
    static_assert(AmbDim >= 1);
    
public:
    
    using Int      = size_t;
    
    using Index_T  = std::array<Int,            DomDim>;
    using Grid_T   = std::array<const Scal_in *,DomDim>;
    
    using Input_T  = std::array<Scal_in, DomDim>;
    using Output_T = std::array<Scal_out,AmbDim>;
    
    static constexpr Int corner_count = (1 << DomDim);
    
protected:
    
    Grid_T  grid;
    Index_T dims;
    const   Scal_out * const      values;
    
    mutable Index_T               i = {0};
    mutable std::array<Input_T,2> t = {{{0},{1}}};
    mutable Input_T               x = {0};
    mutable Output_T              y = {0};
    
public:
    
    Interpolator( const Grid_T & grid_, const Index_T & dims_, const Scal_out * const values_ )
    :   grid(grid_)
    ,   dims(dims_)
    ,   values(values_)
    {}
    
    template< typename I>
    Interpolator( const Scal_in * const * const grid_, const I * const dims_, const Scal_out * const values_ )
    :   values(values_)
    {
        std::copy( &grid_[0], &grid_[DomDim], &grid[0] );
        std::copy( &dims_[0], &dims_[DomDim], &dims[0] );
    }
    
    void Evaluate( const Scal_in * const x_, Scal_out * const y_ )
    {
        Read(x_);
        Find();
        Eval();
        Write(y_);
    }
    
    template<typename I>
    void Evaluate_TensorGrid(
        const Scal_in  * const * const new_grid_,
        const I        * const         new_dims_,
              Scal_out * const         values_
    )
    {
        Grid_T  new_grid;
        Index_T new_dims;

        std::copy( &new_grid_[0], &new_grid_[DomDim], &new_grid[0] );
        std::copy( &new_dims_[0], &new_dims_[DomDim], &new_dims[0] );

        std::array<std::vector<Int>,DomDim> pos;
        std::array<std::vector<Scal_out>,DomDim> t_0;
        std::array<std::vector<Scal_out>,DomDim> t_1;

        for( Int k = 0; k < DomDim; ++k )
        {
            const Scal_in * const g_k = new_grid[k];
            const Int d_k             = new_dims[k];

            std::vector<Int>        pos_k (d_k);
            std::vector<Scal_out>   t_0_k (d_k);
            std::vector<Scal_out>   t_1_k (d_k);

            for( Int j = 0; j < d_k; ++j )
            {
                Find( g_k[j], k );

                pos_k[j] = i[j];
                t_0_k[j] = t[0][j];
                t_1_k[j] = t[1][j];
            }

            std::swap( pos[k], pos_k );
            std::swap( t_0[k], t_0_k );
            std::swap( t_1[k], t_1_k );
        }

        
        dump(ToString(t_0[0]));
        dump(ToString(t_1[0]));

        Index_T idx = {0};

        Int global_idx = 0;
        
        for( Int k = 0; k < DomDim; ++k )
        {
            const Int idx_k = idx[k];
            
            i[k]    = pos[k][idx_k];
            t[0][k] = t_0[k][idx_k];
            t[1][k] = t_1[k][idx_k];
        }

        bool good = true;

        while( good )
        {
            Eval();

            Write( &values_[global_idx] );

            good = Increment( idx, new_dims );
            
            for( Int k = 0; k < DomDim; ++k )
            {
                const Int idx_k = idx[k];
                
                i[k]    = pos[k][idx_k];
                t[0][k] = t_0[k][idx_k];
                t[1][k] = t_1[k][idx_k];
                
                valprint(ToString(k),i[k]);
                valprint(ToString(k),t[0][k]);
                valprint(ToString(k),t[1][k]);
            }

            global_idx += AmbDim;
        }
    }
    
    bool Increment( Index_T & idx, const Index_T & dims_ )
    {
        Int inc = 1;
        
        for( Int k = DomDim; k --> 0; )
        {
            idx[k] += inc;
            
            if( idx[k] >= dims_[k] )
            {
                idx[k] = 0;
                inc = 1;
            }
            else
            {
                return true;
            }
        }
        
        
        return false;
    }
    
    const Index_T & Index() const
    {
        return i;
    }

    Int GlobalIndex( const Index_T & idx ) const
    {
        if constexpr( DomDim == 1 )
        {
            return idx[0];
        }
        else
        {
            Int pos = 0;
            
            for( Int k = 0; k < DomDim - 1; ++k )
            {
                pos = (pos + idx[k]) * dims[k+1];
            }
            
            pos += idx[DomDim-1];

            return pos;
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
        Scal_in volume = static_cast<Scal_in>(1);
        
        for( Int k = 0; k < DomDim; ++k )
        {
            volume *= t[0][k] + t[1][k];
        }
        
        const Scal_in volume_inv = static_cast<Scal_in>(1) / volume;
        
        std::fill( &y[0], &y[AmbDim], static_cast<Scal_out>(0) );
        
        for( Int kdx = 0; kdx < corner_count; ++kdx )
        {
            // For each corner of the grid cuboid, we have to find the global position `pos` of the corner in the grid and its coeffcient `coeff`
        
            Index_T j = i;
            
            Scal_in coeff = volume_inv;
            
            for( Int k = 0; k < DomDim; ++k )
            {
                Int bit = (kdx >> (DomDim-1-k)) & 1;
                j[k] += bit;
                coeff *= t[bit][k];
            }
            
//            dump(i[0]);
//            dump(j[0]);
//            dump(coeff);
            
            const Int pos = AmbDim * GlobalIndex(j);
            
            auto coeff_converted = static_cast<Scal_out>(coeff);
            
            for( Int k = 0; k < AmbDim; ++k )
            {
                y[k] += coeff_converted * values[pos+k];
            }
        }
    }
    
    void Find( const Scal_in z, const Int k )
    {
        const Scal_in * const g = grid[k];
    
        Int a = 0;
        Int b = dims[k]-1;
        
        Scal_in g_a = g[a];
        Scal_in g_b = g[b];
        
        if( z < g_a )
        {
            i[k]    = 0;
            t[1][k] = static_cast<Scal_in>(0);
            t[0][k] = g_a;
            
        }
        else if ( z > g_b )
        {
            i[k]    = b-1;
            t[1][k] = g_b;
            t[0][k] = static_cast<Scal_in>(0);
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
        
            i[k]    = a;
            t[1][k] = z - g_a;
            t[0][k] = g_b - z;
        }
    }
        
    void Find()
    {
        for( Int k = 0; k < DomDim; ++k )
        {
            Find( x[k], k );
        }
    }
    
}; // class Interpolator
