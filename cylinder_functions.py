import numpy as np
from scipy.optimize import brentq
from scipy import interpolate

class BaseState:
    """Optimized state with precomputed values"""
    def __init__(self, R, gr, gt, bc, mu, gMax, set_point, gamma, tau):
        self.R = R                              # Reference radius
        self._Ri = R[0]                         # Inner radius
        self.gr = np.array(gr)                  # Radial growth factor
        self.gt = np.array(gt)                  # Circumferential growth factor
        self.bc = bc                            # Boundary condition
        self.mu = mu                            # Shear modulus     
        self.gMax = gMax                        # Growth factor setpoint
        self.set_point = set_point              # Strain setpoint
        self.gamma = gamma                      # Growth exponent
        self.tau = tau                          # Growth timescale

        #Interpolate gr
        self.gr_interp = interpolate.interp1d(
            self.R, self.gr, kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )

        # Interpolate gt
        self.gt_interp = interpolate.interp1d(
            self.R, self.gt, kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )
    
    @staticmethod
    def entropy(state1, state2, Rs, dt):
        
        strain_energy_1 = np.array([state1.strain_energy_density(state1.find_inner_radius(), R) for R in Rs])
        strain_energy_2 = np.array([state2.strain_energy_density(state2.find_inner_radius(), R) for R in Rs])
        
        int_strain_energy_1 = 2 * np.pi *np.trapezoid(strain_energy_1 * Rs, Rs)
        int_strain_energy_2 = 2 * np.pi *np.trapezoid(strain_energy_2 * Rs, Rs)

        dW = (int_strain_energy_2 - int_strain_energy_1) / dt
        
        return dW

    @staticmethod
    def power_direct(state1, state2, ss, dt):
            
        ri1 = state1.find_inner_radius()
        # Approximate dr/dR using the kinematic relation dr/dR = R/r*g_theta*g_r
        F1_rr = [(s / state1.compute_r(ri1, s)) * state1.gt_interp(s) * state1.gr_interp(s) for s in ss]
        F1_tt = [state1.compute_r(ri1, s) / s for s in ss]


        ri2 = state2.find_inner_radius()
        F2_rr = [(s / state2.compute_r(ri2, s)) * state2.gt_interp(s) * state2.gr_interp(s) for s in ss]
        F2_tt = [state2.compute_r(ri2, s) / s for s in ss]

        dF_rr = [(F2 - F1) / dt for F2, F1 in zip(F2_rr, F1_rr)]
        dF_tt = [(F2 - F1) / dt for F2, F1 in zip(F2_tt, F1_tt)]

        PK1_r = np.array([state1.radial_stress(ri1, s) for s in ss])
        PK1_t = np.array([state1.angular_stress(ri1, s) for s in ss])

        power = PK1_r * dF_rr + PK1_t * dF_tt

        #integrate power
        int_power = 2*np.pi * np.trapezoid(power * ss, ss)  # multiply by R for volume element in cylindrical coordinates

        return int_power

    def compute_r(self, ri, s):
        """Solve r = sqrt(ri^2 + 2*∫s^2*gr*gt^2ds)"""
        if s <= self._Ri:
            return ri
        
        # Simple trapezoidal integration
        x = np.linspace(self._Ri, s, 200)
        integrand = self.gr_interp(x) * self.gt_interp(x) * x
        integral = np.trapezoid(integrand, x)
        return np.sqrt(ri**2 + 2 * integral)
    
    def compute_p(self, ri, s):
        """ Solve p = 
        """

        # Boundary condition for pressure at the inner radius Ri
        gr_Ri = self.gr_interp(self._Ri)
        gt_Ri = self.gt_interp(self._Ri)
        
        # From radial stress boundary condition: sigma_rr(Ri) = bc
        # sigma_rr = (mu/gr)*(s/r)^2 * gt - p
        # bc = (mu/gr_Ri)*(Ri/ri)^2 * gt_Ri - p_i => p_i = (mu/gr_Ri)*(Ri/ri)^2 * gt_Ri - bc
        p_i = self.bc * gr_Ri * gt_Ri - self.mu * (self._Ri / ri)**2 * gt_Ri**2

        if s <= self._Ri:
            return p_i

        # Define integration points
        x = np.linspace(self._Ri, s, 200)
        
        # Pre-calculate values needed for the integral
        r_vals = np.array([self.compute_r(ri, si) for si in x])
        gr_vals = self.gr_interp(x)
        gt_vals = self.gt_interp(x)
        dgr_ds = np.gradient(gr_vals, x)
        dgt_ds = np.gradient(gt_vals, x)
        # dgr_ds, dgt_ds = self._get_derivatives(s)

        # Calculate each term of the integrand
        term1 = 2 * (x / r_vals**2) * (gt_vals / gr_vals)
        term2 = - (x**3 / r_vals**4) * gt_vals**2
        term3 = (x**2 / r_vals**2) / gr_vals * dgt_ds
        term4 = -(x**2 / r_vals**2) * (gt_vals / gr_vals**2) * dgr_ds
        term5 = - 1 / (x * gt_vals**2)
        
        integrand = self.mu * (term1 + term2 + term3 + term4 + term5)
        
        integral = np.trapezoid(integrand, x)

        # Calculate the constant C = p(Ri) / (gr(Ri) * gt(Ri))
        C = p_i / (gr_Ri * gt_Ri)
        
        # Final pressure calculation
        gr_s = self.gr_interp(s)
        gt_s = self.gt_interp(s)
        
        return gr_s * gt_s * (C - integral)
    
    def radial_stress(self, ri, s):
        """Compute radial stress: Pʳᴿ = μ(R²/r²)(gₒ²/gᵣ) + p(r²/(R²gᵣgₒ²))"""
        r_val = self.compute_r(ri, s)
        gr_val = self.gr_interp(s)
        gt_val = self.gt_interp(s)
        p_val = self.compute_p(ri, s)
        
        term1 = self.mu * (s / r_val) * (gt_val / gr_val)
        term2 = p_val * (r_val / s) / (gr_val * gt_val)
        
        return term1 + term2
    
    def angular_stress(self, ri, s):
        r_val = self.compute_r(ri, s)
        gt_val = self.gt_interp(s)
        p_val = self.compute_p(ri, s)
        
        term1 = self.mu * (r_val / s) / (gt_val**2)
        term2 = p_val * (s / r_val)
        
        return term1 + term2

    def radial_cauchy(self, ri, s):
        """Convert PK1 radial stress to Cauchy stress"""
        r_val = self.compute_r(ri, s)
        
        pk1_rr = self.radial_stress(ri, s)
        
        return pk1_rr * (s / r_val)
    
    def hoop_cauchy(self, ri, s):
        """Convert PK1 hoop stress to Cauchy stress"""
        r_val = self.compute_r(ri, s)
        gr_val = self.gr_interp(s)
        gt_val = self.gt_interp(s)
        
        pk1_tt = self.angular_stress(ri, s)
        
        return pk1_tt * (r_val / s) / (gr_val * gt_val)
    
    def find_inner_radius(self):
        """Find inner radius using root finding"""
        def objective(ri):
            return self.radial_stress(ri, 2.0)
        
        # Use a good initial guess based on the previous value
        try:
            return brentq(objective, self._Ri - 0.5, self._Ri + 0.5, 
                         xtol=1e-6, maxiter=20)
        except:
            # If that fails, try a wider bracket
            return brentq(objective, 0.1, 3, xtol=1e-6)
    
    def radial_strain(self, ri, s):
        """A = FG^{-1}"""
        r_val = self.compute_r(ri, s)
        gt_val = self.gt_interp(s)
        return (s / r_val) * gt_val
    
    def hoop_strain(self, ri, s):
        """A = FG^{-1}"""
        r_val = self.compute_r(ri, s)
        gt_val = self.gt_interp(s)
        return r_val / (s * gt_val)
    
    def strain_energy_density(self, ri, s):
        """Compute strain energy density"""
        a_r = self.radial_strain(ri, s)
        a_t = self.hoop_strain(ri, s)

        I1 = a_r**2 + a_t**2

        W = (self.mu / 2) * (I1 - 2)
        
        return W
    
    def Ricci_curvature(self, ri, s):
        """Compute Ricci curvature (not used in LT2)"""

        h = 1e-5  # A small step for finite differences

        gr_s = self.gr_interp(s)
        gt_s = self.gt_interp(s)
        # --- Boundary and Interior Logic ---
        if s == self._Ri:  # Start boundary: use forward differences
            # First derivatives (2nd order forward difference)
            gr_R = (-3*self.gr_interp(s) + 4*self.gr_interp(s+h) - self.gr_interp(s+2*h)) / (2*h)
            gt_R = (-3*self.gt_interp(s) + 4*self.gt_interp(s+h) - self.gt_interp(s+2*h)) / (2*h)
            # Second derivative (1st order forward difference)
            gt_RR = (self.gt_interp(s+2*h) - 2*self.gt_interp(s+h) + self.gt_interp(s)) / (h**2)

        elif s == self.R[-1]:  # End boundary: use backward differences
            # First derivatives (2nd order backward difference)
            gr_R = (3*self.gr_interp(s) - 4*self.gr_interp(s-h) + self.gr_interp(s-2*h)) / (2*h)
            gt_R = (3*self.gt_interp(s) - 4*self.gt_interp(s-h) + self.gt_interp(s-2*h)) / (2*h)
            # Second derivative (1st order backward difference)
            gt_RR = (self.gt_interp(s) - 2*self.gt_interp(s-h) + self.gt_interp(s-2*h)) / (h**2)

        else:  # Interior point: use central differences
            # First derivatives (2nd order central difference)
            gr_R = (self.gr_interp(s+h) - self.gr_interp(s-h)) / (2*h)
            gt_R = (self.gt_interp(s+h) - self.gt_interp(s-h)) / (2*h)
            # Second derivative (2nd order central difference)
            gt_RR = (self.gt_interp(s+h) - 2*gt_s + self.gt_interp(s-h)) / (h**2)

        Ricci = (gr_R * gt_R - gr_s * gt_RR) / (gr_s**3 * gt_s)

        return Ricci

    # def compute_dgt(self, ri, s):
    #     """Compute growth rate based on circumferential stress."""
    #     r_val = self.compute_r(ri, s)
    #     gr_val = self.gr_interp(s)
    #     gt_val = self.gt_interp(s)
    #     p_val = self.compute_p(ri, s)

    #     # This simplifies to dgt = (g_theta / tau) * (1/sigma_star) * (stress_term - sigma_star)
        
    #     stress_term = (self.mu * (r_val / s)**2 / gt_val**2 + p_val) / (gr_val * gt_val)

    #     dgt = self.tau * (stress_term - self.set_point) / self.set_point + 1

    #     return dgt
    
    # def compute_dgr(self, ri, s):
    #     return 1
    
    def update(self):
        """Create updated state"""
        ri = self.find_inner_radius()

        # print gt values
        print("gr min/max:", self.gr.min(), self.gr.max())
        print("gt min/max:", self.gt.min(), self.gt.max())

        # Vectorized dgt computation
        dgt = np.array([self.compute_dgt(ri, s) for s in self.R])
        new_gt = self.gt * dgt
        dgr = np.array([self.compute_dgr(ri, s) for s in self.R])
        new_gr = self.gr * dgr

        return self.__class__(
            self.R, new_gr, new_gt, self.bc, self.mu,
            self.gMax, self.set_point, self.gamma, self.tau
        )
    
    # --- Abstract Methods ---
    # These must be implemented by subclasses.

    def compute_dgt(self, ri, s):
        """The hoop growth law. This is model-specific."""
        raise NotImplementedError("Subclasses must implement the hoop growth law.")

    def compute_dgr(self, ri, s):
        """The radial growth law. This is model-specific."""
        raise NotImplementedError("Subclasses must implement the radial growth law.")
    
    def __str__(self):
        return (f"State(_Ri={self._Ri}, gr={self.gr}, "
                f"gf=[{self.gt[0]:.6f}, {self.gt[-1]:.6f}], "
                f"bc={self.bc}, mu={self.mu})")

