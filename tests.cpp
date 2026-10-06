#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "include/lpt_interp6.h"
#include "include/lpt_microphysics.h"

int main() {
    using TF = double;
    constexpr int n=16;
    std::vector<TF> q(n*n*n);
    // Cell-centered linear field, sampled well away from periodic boundaries.
    for(int k=0;k<n;++k) for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        TF x=(i+0.5)/n, y=(j+0.5)/n, z=(k+0.5)/n;
        q[i+j*n+k*n*n]=1.0+2.0*x-3.0*y+4.0*z;
    }
    Lpt::Periodic_field_view<TF> v{q.data(),n,n,n,0,0,0,n,n*n,1.0/n,1.0/n,1.0/n,0.5,0.5,0.5};
    TF val,gx,gy,gz;
    const TF x=.42,y=.47,z=.51;
    Lpt::sample6_value_gradient(v,x,y,z,val,gx,gy,gz);
    const TF exact=1+2*x-3*y+4*z;
    assert(std::abs(val-exact)<1e-12);
    assert(std::abs(gx-2)<1e-11 && std::abs(gy+3)<1e-11 && std::abs(gz-4)<1e-11);

    Lpt::Growth_config<TF> cfg{1e-10,2.4e-5,997,1.2,1.5e-5,.01,9.81,1.0};
    auto r=Lpt::grow_r2_clamped<TF>(1e-10, 5e-8, -0.5, 10.0, cfg);
    assert(r.r2_new >= 25e-16*(1.0-1e-14));
    auto r2=Lpt::grow_r2_clamped<TF>(26e-16, 5e-8, -0.5, 10.0, cfg);
    assert(std::abs(r2.r2_new-25e-16)<1e-30);
    std::cout << "host interpolation and evaporation clamp tests passed\n";
}
