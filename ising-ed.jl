using LinearAlgebra,LinearMaps
import Arpack
import ArnoldiMethod
import KrylovKit

const L=22
	
diag_ = zeros(Float64,2^L)

function prepareDiag(diag)
	Threads.@threads for state = 1 : 2^L
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
	for i = 1 : L
		Threads.@threads for state = 1 : 2^L
			newstate = (state&(~(2^L))) ⊻ (1<<(i-1))
			if newstate==0
				newstate = 2^L
			end
			C[newstate] -= B[state]
		end
	end
end

println("available threads:",Threads.nthreads())

println("preparing...")
prepareDiag(diag_)

println("computing the lowest eigenvalue...")
H=LinearMap((C,B)->Hfunc!(C,B,diag_),2^L,ismutating=true,issymmetric=true,isposdef=false)

println("using Arpack:")
@time e,v = Arpack.eigs(H,nev=8,which=:SR)
println(e[1])

println("using ArnoldiMethod:")
function eigs_ArnoldiMethod(H)
	decomp,history = ArnoldiMethod.partialschur(H,nev=8,which=ArnoldiMethod.SR())
	e,v = ArnoldiMethod.partialeigen(decomp)
	return e,v
end
@time e,v = eigs_ArnoldiMethod(H)
println(e[1])

println("using KrylovKit:")
function eigs_KrylovKit(H)
	val,vecs,info = KrylovKit.eigsolve(H,rand(eltype(H),size(H,1)),8,:SR;issymmetric=true)
	return val,vecs
end
@time e,v = eigs_KrylovKit(H)
println(e[1])

println("theoretical:")
println(-2sum([ abs(sin((n-1/2) * pi/L)) for n in 1 : L]))
