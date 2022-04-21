# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:26:07 2022

@author: yanisd
"""
from pathlib import Path
import os
import numpy as np
from stl import mesh
import tinyik
from math import sin,cos,asin,acos,atan,sqrt,degrees

#arm = tinyik.Actuator(['z',[1,0.,0.],'z',[1.0,0.,0.]])

# stl converter into solid used by the other classes
class stlfile:
    def __init__(self,solid,unit,location):
        self.solid=solid
    def file(fullName,directory):
        name = "boite"
        directory=Path("C:/Users/yanisd/Documents/stl/")
        fullName=directory/name
    def repair(fullName):
        #https://github.com/WoLpH/numpy-stl/issues/145#issuecomment-702687793
        filenamein="%s.stl"%stlfile.fullName
        filenameout="%s_swapped.stl"%stlfile.fullName

        f1 = open(filenamein, "rb")
        f2 = open(filenameout, "w")

        # Read the header
        header=f1.read(80)
        # Replace if needed
#        header='My new header '.ljust(80)

        # Skip header and read number of triangles
        f1.seek(80, os.SEEK_SET)
        idtype='>u4' # Big Endian uint32

        nt = np.fromfile(f1, dtype=idtype,count=1)[0]

        # Read triangles one by one. Use big endian.
        fdtype='>f4' # Big Endian float32

        normals=np.zeros((nt,3),dtype='<f4')
        v1=np.zeros((nt,3),dtype='<f4')
        v2=np.zeros((nt,3),dtype='<f4')
        v3=np.zeros((nt,3),dtype='<f4')
        attr=np.zeros(nt,dtype='<u2')
        for i in range(nt):
            normals[i,:] = np.fromfile(f1, dtype=fdtype,count=3)
            v1[i,:] = np.fromfile(f1, dtype=fdtype,count=3)
            v2[i,:] = np.fromfile(f1, dtype=fdtype,count=3)
            v3[i,:] = np.fromfile(f1, dtype=fdtype,count=3)
            attr[i] = np.fromfile(f1, dtype='>u2',count=1)
        f1.close()
        f2.write(header)
        np.asarray(nt,dtype='<u4').tofile(f2)
        for i in range(nt):
            normals[i,:].tofile(f2)
            v1[i,:].tofile(f2)
            v2[i,:].tofile(f2)
            v3[i,:].tofile(f2)
            attr[i].tofile(f2)
        f2.close()
    def mesh(my_mesh):
        my_mesh = mesh.Mesh.from_file('%s.stl' %(stlfile.fullName))
        return my_mesh
    def properties(volume,cog,inertia):
        volume, cog, inertia = stlfile.my_mesh.get_mass_properties()
        print("Volume                                  = {0}".format(volume))
        print("Position of the center of gravity (COG) = {0}".format(cog))
        print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
        print("                                          {0}".format(inertia[1,:]))
        print("                                          {0}".format(inertia[2,:]))

class iksolver :
    def __init__(self, center, radius):
        self.center=center
        self.radius=radius
        
    def cvxopt_solve_minmax(n, a, B, x_min=-42, x_max=42, solver=None):
        c = np.hstack([np.zeros(n), [1]])
        # cvxopt constraint format: G * x <= h
        # first,  a + B * x[0:n] <= x[n]
        G1 = np.zeros((n, n + 1))
        G1[0:n, 0:n] = B
        G1[:, n] = -np.ones(n)
        h1 = -a
    
        # then, x_min <= x <= x_max
        x_min = x_min * np.ones(n)
        x_max = x_max * np.ones(n)
        G2 = np.vstack([
            np.hstack([+np.eye(n), np.zeros((n, 1))]),
            np.hstack([-np.eye(n), np.zeros((n, 1))])])
        h2 = np.hstack([x_max, -x_min])
    
        c = np.cvxopt.matrix(c)
        G = np.cvxopt.matrix(np.vstack([G1, G2]))
        h = np.cvxopt.matrix(np.hstack([h1, h2]))
        sol = np.cvxopt.solvers.lp(c, G, h, solver=solver)
        return np.array(sol['x']).reshape((n + 1,))

    def coordinate_to_degrees(x, y): # function to convert coordinates to angles from the x-axis (0~360)
        x += 0.00001 # this is to avoid zero division error in case x == 0
     
        if x >= 0 and y >= 0:   # first quadrant
            angle = degrees(atan(y/x))
        elif x < 0 and y >= 0:  # second quadrant
            angle = 180 + degrees(atan(y/x))
        elif x < 0 and y < 0:   # third quadrant
            angle = 180 + degrees(atan(y/x))
        elif x >= 0 and y < 0:  # forth quadrant
            angle = 360 + degrees(atan(y/x))
        return round(angle,1)


class Circle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def intersect(self, other):
        dist = np.linalg.norm(self.center - other.center)
        if dist > self.radius + other.radius:
    #no contact
            return None
        elif dist == 0 and self.radius == other.radius:
    #Coincident
            return np.inf
        elif dist + min(self.radius, other.radius) < max(self.radius, other.radius):
    #contained
            return None
        else:
    #two intersections
            a = ((self.radius**2 - other.radius**2) + dist**2) / (2*dist)
            h = np.sqrt(self.radius**2-a**2)
            p2=self.center + (a*(other.center-self.center))
#            i1[0] += h* (other.center[1] - self.center[1] / dist)
#shoulder_theta = sigangle(self.shoulder_offset, [0,1])
class anticollision:
#GJK anticolision algorithm basis ♥ with Minkowski sum to determine if collision point exist between 2 polygons and simplify concave polygons in simplex triangles
    ORIGIN = [0,0,0]
    def __init__(self,s1, s2, d):
        self.s1=s1
        self.s2=s2
        self.d=d
    def GJK(s1,s2):
        #True if shapes s1 and s2 intersect
        #all vectors/points are "3d" ([x,y,0])
        d=np.normalize(s2.center - s1.center)
        simplex = [anticollision.support(s1,s2,d)]
        d=anticollision.ORIGIN - simplex[0]
        while True:
            A= anticollision.support(s1,s2,d)
            if np.dot(A,d) < 0:
                return False
            simplex.append(A)
            if anticollision.handleSimplex(simplex,d):
                return True
    def support(s1,s2,d):#determine furthest point of the simplex to identify the simplex
        return s1.furthestPoint(d) - s2.furthestPoint(-d)
    def handleSimplex(simplex, d):#determine if the simplex need to bee solved with lineCase or triangleCase
        if len(simplex)==2 :
               return anticollision.lineCase(simplex,d)
        return anticollision.TriangleCase(simplex,d)
    def lineCase(simplex, d):
        B,A=simplex
        AB,OA=B-A,anticollision.ORIGIN-A
        ABperp=np.tripleProd(AB,OA,AB)
        d.set(ABperp)
        return False
    def triangleCase(simplex,d):
        C, B, A = simplex
        AB, AC, AO = B-A,C-A,anticollision.ORIGIN-A
        ABperp = np.tripleProd(AC,AB,AB)
        ACperp = np.tripleProd(AB,AC,AC)
        if np.dot(ABperp, AO)>0:#region AB
            simplex.remove(C);d.set(ABperp)
            return False
        elif np.dot(ACperp,AO) > 0:#region AC
            simplex.remove(B); d.set(ACperp)
            return False
        return True
    def sigangle(A, B):
        ANorm = np.normalize(A)
        BNorm = np.normalize(B)
        dot = np.dot(ANorm, BNorm)
        clippedDot = np.clip(dot, -1.0, 1.0)
        ang = np.accross(clippedDot)
        
        perp = [BNorm[1], -BNorm[0]]
        
        if np.dot(ANorm, perp) < 0:
            return -ang
        else:
            return ang
