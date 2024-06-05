import math as m 
import numpy as np
import sympy as s 
import matplotlib.pyplot as plt


class EinsteinFieldEquations:
    def __init__(self,M):
        self.G = s.symbols("G")
        self.M = s.symbols("M")
        self.c = s.symbols("c")

    def kruskal_szekeres_metric(self):
        T,X,theta,phi,r = s.symbols("T X theta phi r")
        coeffs = [T,X,theta,phi]
        g_00 = -(32 * (self.G ** 3) * (self.M ** 3) * 1/r) * s.exp(-r/(2*self.G*self.M))
        g_11 = (32 * (self.G ** 3) * (self.M ** 3) * 1/r) * s.exp(-r/(2*self.G*self.M))
        g_22 = r ** 2
        g_33 = (r ** 2) * s.sin(theta)*s.sin(theta)
        g_munu = [[g_00,0,0,0],[0,g_11,0,0],[0,0,g_22,0],[0,0,0,g_33]]
        return g_munu, coeffs
        
    def minkowski_metric(self):
        t,x,y,z = s.symbols("t x y z")
        coeffs = [t,x,y,z]
        g_munu = [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        return g_munu, coeffs

    def schwarzchild_metric(self):
        t,r,theta,phi = s.symbols("t r theta phi")
        coeffs = [t,r,theta,phi]
        rs = (2*self.G*self.M)/(self.c * self.c)
        g_00 = -(1-(rs/r))*self.c*self.c 
        g_11 = (1-(rs/r))**(-1)
        g_22 = r*r 
        g_33 = r*r*s.sin(theta)*s.sin(theta)
        g_munu = [[g_00,0,0,0],[0,g_11,0,0],[0,0,g_22,0],[0,0,0,g_33]]
        return g_munu, coeffs

    def first_order_conn(self,i,j,k,g_munu):
        g_ijk = -s.diff(g_munu[i][j],coeffs[k])
        g_jki = s.diff(g_munu[j][k],coeffs[i])
        g_kij = s.diff(g_munu[k][i],coeffs[j])
        return 1/2 * (g_ijk + g_jki + g_kij)

    def christoffel_symbols(self,g_munu,coeffs):
        inv_g_munu = np.array(s.Matrix(g_munu).inv())         
        christoffel_matrix = [[[0 for i in range(4)] for j in range(4)] for k in range(4)] 
        non_zero_indices = []
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    second_order_christoffel = 0
                    for r in range(4):
                        first_order_christoffel = self.first_order_conn(j,k,r,g_munu)
                        second_order_christoffel += inv_g_munu[i][r] * first_order_christoffel
                        
                    christoffel_matrix[i][j][k] = second_order_christoffel
                    
                    if christoffel_matrix[i][j][k] != 0:
                        non_zero_indices.append((i,j,k))
        return christoffel_matrix,non_zero_indices

    def ricci_tensor(self,metric,coeffs):
        non_zero_conns, indices = self.christoffel_symbols(metric,coeffs)
        ricci_curvature = [[0 for a in range(4)] for b in range(4)]
        for i in range(4):
            for j in range(4):
                R1 = sum([s.diff(non_zero_conns[a][i][j],coeffs[a]) for a in range(4)])
                R2 = sum([s.diff(non_zero_conns[a][a][i],coeffs[j]) for a in range(4)])
                R3 = sum([sum([(non_zero_conns[a][a][b]*non_zero_conns[b][i][j] - non_zero_conns[a][i][b]*non_zero_conns[b][a][j]) for a in range(4)]) for b in range(4)])
                R_ij = s.simplify(R1 - R2 + R3)
                ricci_curvature[i][j] = R_ij
                
        return ricci_curvature
    
    def scalar_curvature(self,metric,coeffs):
        inv_g_munu = np.array(s.Matrix(metric).inv())
        Rmunu = self.ricci_tensor(metric,coeffs)
        R = sum([sum([inv_g_munu[i][j] * Rmunu[i][j] for i in range(4)]) for j in range(4)])
        return R
    
    def einstein_tensor(self,metric,coeffs):
        Gmunu = [[0 for i in range(4)] for j in range(4)]
        Rmunu = self.ricci_tensor(metric,coeffs)
        R = self.scalar_curvature(metric,coeffs)
        for mu in range(4):
            for nu in range(4):
                Gmunu[mu][nu] = Rmunu[mu][nu] - 1/2*metric[mu][nu]*R
        return Gmunu
    
    def geodesic_equation(self,christoffel_matrix,indices,coeffs):
        equations = [0 for i in range(4)]
        tau = s.symbols("tau")
        for mu in range(4):
            dxmu2ds2 = s.diff(s.diff(coeffs[mu],tau),tau)
            right_side = sum([sum([christoffel_matrix[mu][alpha][beta]*s.diff(coeffs[alpha],tau)*s.diff(coeffs[beta],tau) for alpha in range(4)]) for beta in range(4)])
            equations[mu] = dxmu2ds2 + right_side
        return equations
    
    def display_christoffels(self,christoffel_matrix,indicies):
        for i in indicies:
            display(Latex((str(i) + ": " + str(christoffle_matrix[i[0]][i[1]][i[2]]))))
            
    def display_ricci(self,ricci_tensor):
        print(ricci_tensor)
        
    def display_einstein_tensor(self,einstein_tensor):
        print(einstein_tensor)



mass = 2 * 10**31
EFE = EinsteinFieldEquations(mass)
a,b = EFE.schwarzchild_metric()
EFE.christoffel_symbols(a,b)
