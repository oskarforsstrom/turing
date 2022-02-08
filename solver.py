

# Grierer-Meinhardt reaction functions
def GM_f(u,v, k=0, c1=0, c2=0, c3=0):
    return c1 - c2*u + c3*( u**2 / ((1 + k*u**2)*v) )

def GM_g(u,v, c4=0, c5=0):
    return c4*u**2 -c5*v

