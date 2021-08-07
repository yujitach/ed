using LinearAlgebra,LinearMaps
import Arpack

const L=18
	
diag_ = zeros(Float64,2^L)

function prepareDiag(diag)
	for state = 1 : 2^L
		for i = 1 : L
			j = i==L ? 1 : i+1
			diag[state] -= (((state >> (i-1))&1) == ((state >> (j-1))&1)) ? 1 : -1
		end
	end
end
	
function Hfunc!(C,B,diag)
	for state = 1 : 2^L
		C[state] = diag[state] * B[state]
	end
	for state = 1 : 2^L
		for i = 1 : L
			newstate = (state&(~(2^L))) ⊻ (1<<(i-1))
			if newstate==0
				newstate = 2^L
			end
			C[newstate] -= B[state]
		end
	end
end


println("preparing...")
prepareDiag(diag_)

println("computing the lowest eigenvalue...")
H=LinearMap((C,B)->Hfunc!(C,B,diag_),2^L,ismutating=true,issymmetric=true,isposdef=false)
@time e,v = Arpack.eigs(H,nev=1,which=:SR)

println("obtained:")
println(e[1])

println("theoretical:")
println(-2sum([ abs(sin((n-1/2) * pi/L)) for n in 1 : L]))