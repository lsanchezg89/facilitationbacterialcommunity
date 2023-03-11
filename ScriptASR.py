import numpy as np


#Lotka-Volterra parameters
alphasa = 1.65
alphaas = 2.0

alphasr = 0.6
alphars = 0.4

alphaar = 0.5
alphara = 0.6

#Antagonistic strain parameters
Da = 1.7e-7
ra =0.0078

#Antagonistic strain Hill function parameters
qa = 3.5
Ka = 1.0e-8

#Sensitive strain parameters
Ds = 1.02e-7
rs = 0.0086

#Sensitive strain Hill function parameters
qs = 3.5
Ks = 1.0e-14

#Resistant strain parameters
Dr = 5.1e-8  
rr = 0.0047

#Resistant strain Hill function parameters
qr = 3.
Kr =1.0e-28

#Metabolite parameters
Dm = 0.34
rm = .2
gammam = 0.01

#Antagonistic substance parameters
ru = 0.2
qu = 3.5
Ku = 1.0e-12
gammau = 0.01
Du = 0.1615

#Hill function bounding parameter (parameter d in model)
f = 0.25 

#Simulation parameters
sizex = 60
sizey = sizex
ratiodiag = 1
ratioside = 4
dx = .1
T = 28800
dt = .01

#Bacteria initial conditions

#Antagonistic strain
def InitConditionsLeft(X):
	for i in range (0, sizex):
		for j in range (0, sizey):
			X[i,j] = np.random.normal(0.02,0.02*0.1)			
	return X

#Sensitive strain
def InitConditionsRight(X):
	for i in range (0, sizex):
		for j in range (0, sizey):
			X[i,j] = np.random.normal(0.2,0.2*0.1)
	return X

#Resistant strain	
def InitConditionsCenter(X):
	for i in range(0, sizex):
		for j in range(0, sizey):
			X[i,j] = np.random.normal(1.0,1.0*0.1)
	return X

#Laplacian

def laplacianmiddle(Z):
    	Ztop = Z[0:-2, 1:-1]
    	Zleft = Z[1:-1, 0:-2]
    	Zbottom = Z[2:, 1:-1]
    	Zright = Z[1:-1, 2:]
    	Zcenter = Z[1:-1, 1:-1]
    	Ztopleft = Z[0:-2,0:-2]
    	Zbottomleft = Z[2:,0:-2]
    	Ztopright = Z[0:-2,2:]
    	Zbottomright = Z[2:,2:]
    	return (ratioside * (Ztop + Zleft + Zbottom + Zright) 
    		+ ratiodiag * (Ztopleft + Zbottomleft + Ztopright + Zbottomright) -
    		4 * (ratioside + ratiodiag) * Zcenter) / (6*dx**2)

def laplacianupperboundary(Z):
	Zcenter = Z[0, 1:-1]
	Zbottom = Z[1, 1:-1]
	Zleft = Z[0, 0:-2]
	Zright = Z[0, 2:]
	Zbottomleft = Z[1, 0:-2]
	Zbottomright = Z[1, 2:]
	return (ratioside * (Zleft + Zbottom + Zright)
		+ ratiodiag * (Zbottomleft + Zbottomright)
		- 4 * (ratioside + ratiodiag) * Zcenter) / (6*dx**2)

def laplacianlowerboundary(Z):
	Zcenter = Z[-1, 1:-1]
	Ztop = Z[-2, 1:-1]
	Zleft = Z[-1,0:-2]
	Zright = Z[-1, 2:]
	Ztopleft = Z[-2, 0:-2]
	Ztopright = Z[-2, 2:]
	return (ratioside * (Zleft + Ztop + Zright)
		+ ratiodiag * (Ztopleft + Ztopright)
		- 4 * (ratioside + ratiodiag) * Zcenter) / (6*dx**2)
		
def laplacianleftboundary(Z):
	Zcenter = Z[1:-1, 0]
	Ztop = Z[1:-1, 1]
	Zleft = Z[0:-2, 0]
	Zright = Z[2:, 0]
	Ztopleft = Z[0:-2, 1]
	Ztopright = Z[2:, 1]
	return (ratioside * (Zleft + Ztop + Zright)
		+ ratiodiag * (Ztopleft + Ztopright)
		- 4 * (ratioside + ratiodiag) * Zcenter) / (6*dx**2)

def laplacianrightboundary(Z):
	Zcenter = Z[1:-1, -1]
	Zbottom = Z[1:-1, -2]
	Zleft = Z[0:-2, -1]
	Zright = Z[2:, -1]
	Zbottomleft = Z[0:-2, -2]
	Zbottomright = Z[2:, -2]
	return (ratioside * (Zleft + Zbottom + Zright)
		+ ratiodiag * (Zbottomleft + Zbottomright)
		- 4 * (ratioside + ratiodiag) * Zcenter) / (6*dx**2)
	

def laplacian(Y):
	DeltaY = np.zeros((sizex,sizey))
	DeltaY[0,0] = (ratioside * (Y[0,1] + Y[1,0]) + ratiodiag * Y[1,1] - 4 * (ratioside + ratiodiag) * Y[0,0]) / (6*dx**2)
	DeltaY[-1,0] = (ratioside * (Y[-2,0] + Y[-1,1]) + ratiodiag * Y[-2,1] - 4 * (ratioside + ratiodiag) * Y[-1,0]) / (6*dx**2)
	DeltaY[-1,-1] = (ratioside * (Y[-2,-1] + Y[-1,-2]) + ratiodiag * Y[-2,-2] - 4 * (ratioside + ratiodiag) * Y[-1,-1]) / (6*dx**2)
	DeltaY[0,-1] = (ratioside * (Y[0,-2] + Y[1,-1]) + ratiodiag * Y[1,-2] - 4 * (ratioside + ratiodiag) * Y[0,-1]) / (6*dx**2)
	DeltaY[1:-1,1:-1] = laplacianmiddle(Y)
	DeltaY[0, 1:-1] = laplacianupperboundary(Y)
	DeltaY[-1, 1:-1] = laplacianlowerboundary(Y)
	DeltaY[1:-1, 0] = laplacianleftboundary(Y)
	DeltaY[1:-1, -1] = laplacianrightboundary(Y)
	return DeltaY

#Model

A0 = np.zeros((sizex, sizey))
S0 = np.zeros((sizex, sizey))
R0 = np.zeros((sizex, sizey))
A = np.zeros((sizex, sizey, 101))
S = np.zeros((sizex, sizey, 101))
R = np.zeros((sizex, sizey, 101))
m = np.zeros((sizex, sizey, 101))
u = np.zeros((sizex, sizey, 101))

def Integrate(A,S,R,m,u,j):
	A[:,:,0] = At[:,:,j-1]
	S[:,:,0] = St[:,:,j-1]
	R[:,:,0] = Rt[:,:,j-1]
	for i in range(0,100):
		deltaA = laplacian(A[:,:,i])
		deltaS = laplacian(S[:,:,i])
		deltaR = laplacian(R[:,:,i])
		deltam = laplacian(m[:,:,i])
		deltau = laplacian(u[:,:,i])
		newA = A[:,:,i]
		newS = S[:,:,i]
		newR = R[:,:,i]
		newm = m[:,:,i]
		newu = u[:,:,i]
		A[:,:,i+1] = newA + dt * (Da * deltaA + ra * (f * (Ka**qa / (newm**qa + Ka**qa)) + (1 - f)) * (1 - newA - alphaas * newS - alphaar * newR) * newA)
		S[:,:,i+1] = newS + dt * (Ds * deltaS + rs * (Ks**qs / (newu**qs + Ks**qs)) * (1 - newS - alphasa * newA - alphasr * newR) * newS)
		R[:,:,i+1] = newR + dt * (Dr * deltaR + rr * (f * (Kr**qr / (newu**qr + Kr**qr)) + (1 - f)) * (1 - newR - alphara * newA - alphars * newS) * newR)
		m[:,:,i+1] = newm + dt * (Dm * deltam + rm * (newS + newR) - gammam * newm)
		u[:,:,i+1] = newu + dt * (Du * deltau + ru * (newm**qu / (newm**qu + Ku**qu)) * newA - gammau * newu)
	return A[:,:,-1], S[:,:,-1], R[:,:,-1], m[:,:,-1], u[:,:,-1]

At0 = InitConditionsLeft(A0)
St0 = InitConditionsRight(S0)
Rt0 = InitConditionsCenter(R0)

At = np.zeros((sizex, sizey, T+1))
St = np.zeros((sizex, sizey, T+1))
Rt = np.zeros((sizex, sizey, T+1))
mt = np.zeros((sizex, sizey, T+1))
ut = np.zeros((sizex, sizey, T+1))

At[:,:,0] = At0
St[:,:,0] = St0
Rt[:,:,0] = Rt0
for j in range(1,T+1):
	At[:,:,j], St[:,:,j], Rt[:,:,j], mt[:,:,j], ut[:,:,j] = Integrate(A,S,R,m,u,j)

    
np.save("A.npy", At)
np.save("S.npy", St)
np.save("R.npy", Rt)

