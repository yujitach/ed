using LinearAlgebra,LinearMaps
using SparseArrays
using Arpack
using ArnoldiMethod
using KrylovKit
using CUDA
using CUDA.CUSPARSE

const L = 6
const nev = 1

println()
println("exact diagonalization of L=", L, " keeping nev=", nev)
println()

#=

NOTE To facilitate the code, I always specify the variable types for functions.

The following types are used:
	Bool for anything binary contained in a vector.
		e.g. fusion flag where 0 indicates allowed and 1 disallowed.
	Int8 for anything constant or linear in L contained in a vector.
		e.g. i, pos.
	Int64 for everything else, including variables exponential in L.
		e.g. state, ind, flag, edgeState.
		edgeState assigns one byte for each site, so Int128 needed if L>21.

=#

#=

As Julia's array is 1-based, I use

	"ind" = 1 ... 4^L

as the array index.

"state" is what you obtain by encoding vertex labels using the following mapping,
and takes the values 0 ...  4^L-1.

=#

# y, z, p, m are x, 0, +, - but with ρ draped below at that vertex
const mapping=Dict('x'=>0, '0'=>1, '+'=>2, '-'=>3, 'y'=>0, 'z'=>1, 'p'=>2, 'm'=>3)
const revmapping=Dict(0=>'x', 1=>'0', 2=>'+', 3=>'-')

const startMapping = Dict('0'=>0, '1'=>1, '2'=>2)

#=

The conversion between "ind" and "state" is basically done by considering modulo 4^L.

When L is even, however, some additional care is needed,
since "xxx...x" can come with two "start labels".
My convention is to use
	ind == 4^L  for "x....x" with ρ as the start label
	ind == 1  for "x....x" with 1 as the start label

NOTE For use in zipper, state has two extra bits to encode the start label,
and ~log(L) extra bits to encode the position where ρ is draped below.
	e.g. "x....x" with aρ as the start label and where third x is draped below
		has state with bitstring 0...011010....0 meaning
		0...011   01   0....0
		draped  start  x....x

Now state has two meanings, either with the start/draped info or without.
TODO It is better to use different variable names to distinguish the two.
NOTE I have also decided to change the relation between ind and state to
	ind = state + 1
so that it is less awkward incorporating the extra bits for start/draped.

=#

function stateFromString(s::String,L::Int64=L)
	if length(s)!=L && length(s)!=L+2
		error("length doesn't match")
	end
	if length(s)==L
		s = s*"_0"
	end
	start=startMapping[s[L+2]]
	a=start<<(2*L)
	for i in L : -1 : 1
		if s[i] in ['y','z','p','m']
			a+=(i<<(2*(L+1)))
		end
		a+=(mapping[s[i]]<<(2*(i-1)))
	end
	return a
end

function stringFromState(state::Int64,L::Int64=L)
	below = (state>>(2*(L+1)))
	start = (state>>(2*L)) & 3
	if iseven(L) && (state&(4^L-1))==1
		state -= 1
	end
	s=""
	for i in 1 : L
		if i == below
			if (state&3) == 0
				s*='y'
			elseif (state&3) == 1
				s*='z'
			elseif (state&3) == 2
				s*='p'
			elseif (state&3) == 3
				s*='m'
			else
				error("no match")
			end
		else
			s*=revmapping[(state&3)]
		end
		state>>=2
	end
	s*='_'
	s*=string(start)
	return s
end


#=

Divide-and-conquer stuff.

=#

kantaro_ = Dict{Tuple{Int64,Bool,Bool,Int64},Vector{Int64}}()

for el = 0 : 1
	for er = 0 : 1
		for q = 0 : 2
			kantaro_[(1, el, er, q)] = []
		end
	end
end
kantaro_[(1, true, true, 0)] = [ 1 ]
kantaro_[(1, true, true, 1)] = [ 2 ]
kantaro_[(1, true, true, 2)] = [ 3 ]

function setKantaro!(kantaro::Dict{Tuple{Int64,Bool,Bool,Int64},Vector{Int64}},
	L::Int64, evenxs_left::Bool, evenxs_right::Bool, q::Int64)
	kantaro[(L, evenxs_left, evenxs_right, q)] = []
	if evenxs_right
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], getKantaro(kantaro, L-1, evenxs_left, false, q) ) # append x
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], [ x + (1 << (2*(L-1))) for x in getKantaro(kantaro, L-1, evenxs_left, true, q) ] )
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], [ x + (2 << (2*(L-1))) for x in getKantaro(kantaro, L-1, evenxs_left, true, (q+2)%3) ] )
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], [ x + (3 << (2*(L-1))) for x in getKantaro(kantaro, L-1, evenxs_left, true, (q+1)%3) ] )
	else
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], getKantaro(kantaro, L-1, evenxs_left, true, q) )
	end
	if ((iseven(L) && !evenxs_left) || (isodd(L) && evenxs_left)) && evenxs_right
		append!( kantaro[(L, evenxs_left, evenxs_right, q)], [ (q+1) << (2*(L-1)) ] )
	end
end

function getKantaro(kantaro::Dict{Tuple{Int64,Bool,Bool,Int64},Vector{Int64}},
	L::Int64, evenxs_left::Bool, evenxs_right::Bool, q::Int64)::Vector{Int64}
	if !haskey(kantaro, (L, evenxs_left, evenxs_right, q))
		setKantaro!(kantaro, L, evenxs_left, evenxs_right, q)
	end
	return kantaro[(L, evenxs_left, evenxs_right, q)]
end

function getKantaro(kantaro::Dict{Tuple{Int64,Bool,Bool,Int64},Vector{Int64}}, L::Int64)::Vector{Int64}
	res = []
	append!(res, getKantaro(kantaro_, L, true, true, 0))
	append!(res, getKantaro(kantaro_, L, false, false, 0))
	if iseven(L)
		append!(res, [0, 1])
	end
	sort!(res)
	return res
end

println("computing basis...")
@time const basis = [ x+1 for x in getKantaro(kantaro_, L) ]
const len = length(basis)
const fromInd = Dict((basis[x],x) for x in 1:len)
println()


#=
TODO trailingXs is only used for inferring whether start label is type 1 or ρ.
Might as well define a function that also checks whether state is 1 when L even.
=#
function trailingXs(state::Int64,L::Int64=L)
	state = state & (4^L-1)
	i=0
	while state!=0
		state>>=2
		i+=1
	end
	return L-i
end

#=

Modified to allow for nontrivial start/draped used in the zipper.

But zipper only cares about the main flag, which is computed by setFusionFlag!,
so might as well keep Yuji's original definition.

=#

# Changed to Int64 since Int32 will not be enough even for L=15.
flag_ = zeros(Int64,len)

function setFlag!(flag::Vector{Int64},preind::Int64,L::Int64=L)
	ind = basis[preind]
	flagshift=L+2
	state=ind-1
	below=(state>>(2*(L+1)))
	start=(state>>(2*L))&3
	state=state&(4^L-1)
	if (start==3) || (below>=L)
		flag[preind] |= 3 << flagshift
		return
	end
	if state==0 && isodd(L)
		flag[preind] |= 3 << flagshift
		return
	end
	evenxs=iseven(trailingXs(state,L))
	tot=start
	if(state==1 && iseven(L))
		state=0
		evenxs=false
	end
	for pos = 0 : L-1
		tot%=3
		a=(state >> (2*pos)) & 3
		if a==0
			if(evenxs)
				flag[preind] |= 1<<pos
			end
			evenxs = ! evenxs
			if ((pos+1)==below || (below!=0 && pos==L-1))
				tot = 3-tot
			end
		else
			if(!evenxs)
				# not allowed
				flag[preind] |= 3 << flagshift
				return
			end
			if a==2
				tot+=1
			elseif a==3
				tot+=2
			end
		end
	end
	tot=tot-start
	if tot<0
		tot+=3
	end
	tot%=3
	if state==0 && isodd(L)
		flag[preind] |= 3 << flagshift
		return
	end
	flag[preind] |= tot << flagshift
end

diag_ = zeros(Float64,len)

const U = 10	# suppression factor for forbidden state
const Z = 10	# suppression factor for states charged under twisted Z3
const ζ = (√(13)+3)/2
const ξ = 1/√ζ
const x = (2-√(13))/3
const z = (1+√(13))/6
const y1 = (5-√(13) - √(6+6*√(13)))/12
const y2 = (5-√(13) + √(6+6*√(13)))/12

const sXX=(0,0)
const sX0=(0,1)
const sXP=(0,2)
const sXM=(0,3)
const s0X=(1,0)
const s00=(1,1)
const s0P=(1,2)
const s0M=(1,3)
const sPX=(2,0)
const sP0=(2,1)
const sPP=(2,2)
const sPM=(2,3)
const sMX=(3,0)
const sM0=(3,1)
const sMP=(3,2)
const sMM=(3,3)

function stateFromInd(ind::Int64,L::Int64=L)
	state=(ind-1)&(4^L-1)
	if state==1 && iseven(L)
		state=0
	end
	return state
end

function mainFlag(flag::Vector{Int64},preind::Int64,L::Int64=L)::Int8
	flagshift=L+2
	return (flag[preind] >> flagshift) & 3
end

function nextSite(i::Int64,L::Int64=L)
	j=i+1
	if j==L+1
		j=1
	end
	return j
end

function localStatePair(state::Int64,i::Int64,L::Int64=L)
	j = nextSite(i,L)
	a = (state >> (2*(i-1))) & 3
	b = (state >> (2*(j-1))) & 3
	return a,b
end

function isρ1ρ(flag::Vector{Int64},preind::Int64,i::Int64)
	return ((flag[preind] >> (i-1)) & 1) == 1
end

function computeDiag!(diag::Vector{Float64},flag::Vector{Int64},preind::Int64)
	ind = basis[preind]
	state=stateFromInd(ind)
	fl = mainFlag(flag,preind)
	if fl==3
		diag[preind]=U
		return
	elseif fl==1 || fl==2
		diag[preind]=Z
		return
	end
	diag[preind]=0
	for i = 1 : L
		sp=localStatePair(state,i)
		if sp==sX0
			diag[preind] -= 1
		elseif sp==s0X
			diag[preind] -= 1
		elseif sp==sXX && isρ1ρ(flag,preind,i)
			diag[preind] -= 1/ζ
		elseif sp==sPM
			diag[preind] -= y1 * y1
		elseif sp==sMP
			diag[preind] -= y2 * y2
		elseif sp==s00
			diag[preind] -= x * x
		elseif sp==s0P
			diag[preind] -= y1 * y1
		elseif sp==sP0
			diag[preind] -= y2 * y2
		elseif sp==sMM
			diag[preind] -= z * z
		elseif sp==s0M
			diag[preind] -= y2 * y2
		elseif sp==sM0
			diag[preind] -= y1 * y1
		elseif sp==sPP
			diag[preind] -= z * z
		end
	end
end

function newInd(state::Int64,i::Int64,sp::Tuple{Int64, Int64})
	(a,b)=sp
	state &= ~(3<<(2*(i-1)))
	state |= (a<<(2*(i-1)))
	j=nextSite(i)
	state &= ~(3<<(2*(j-1)))
	state |= (b<<(2*(j-1)))
	if(state!=0)
		return 1+state
	end
	if(isodd(i))
		return 1
	else
		return 2
	end
end

println("available number of threads: ", Threads.nthreads())
println()
println("preparing...")
Threads.@threads for preind = 1 : len
	setFlag!(flag_,preind)
	computeDiag!(diag_,flag_,preind)
end
println()

newPreind(state,i,sp) = fromInd[newInd(state,i,sp)]

function buildH(diag,flag)
	res = sparse(Int64[],Int64[],Float64[],len,len)
	col=Int64[]
	row=Int64[]
	val=Float64[]
	ncol = 1
	for preind = 1 : len
		ind = basis[preind]
		state=stateFromInd(ind)
		# cannot directly append into row, val because row needs to be ordered
		# define minirow/val and append to row/val after sorting
		minirow = [preind]
		minival = [diag[preind]]
		for i = 1 : L
			sp=localStatePair(state,i)
			if sp==sXX  && isρ1ρ(flag,preind,i)
				append!(minirow,map(s->newPreind(state,i,s),[sPM,sMP,s00]))
				append!(minival,-ξ .* [y1,y2,x])
			elseif sp==sPM
				append!(minirow,map(s->newPreind(state,i,s),[sXX,sMP,s00]))
				append!(minival,-y1 .* [ξ,y2,x])
			elseif sp==sMP
				append!(minirow,map(s->newPreind(state,i,s),[sXX,sPM,s00]))
				append!(minival,-y2 .* [ξ,y1,x])
			elseif sp==s00
				append!(minirow,map(s->newPreind(state,i,s),[sXX,sPM,sMP]))
				append!(minival,-x .* [ξ,y1,y2])
			elseif sp==s0P
				append!(minirow,map(s->newPreind(state,i,s),[sP0,sMM]))
				append!(minival,-y1 .* [y2,z])
			elseif sp==sP0
				append!(minirow,map(s->newPreind(state,i,s),[s0P,sMM]))
				append!(minival,-y2 .* [y1,z])
			elseif sp==sMM
				append!(minirow,map(s->newPreind(state,i,s),[s0P,sP0]))
				append!(minival,-z .* [y1,y2])
			elseif sp==s0M
				append!(minirow,map(s->newPreind(state,i,s),[sM0,sPP]))
				append!(minival,-y2 .* [y1,z])
			elseif sp==sM0
				append!(minirow,map(s->newPreind(state,i,s),[s0M,sPP]))
				append!(minival,-y1 .* [y2,z])
			elseif sp==sPP
				append!(minirow,map(s->newPreind(state,i,s),[s0M,sM0]))
				append!(minival,-z .* [y2,y1])
			end
		end

		# TODO define helper function
		# need to sort rows and also add values of coinciding rows
		perm = sortperm(minirow)
		minirow = minirow[perm]
		minival = minival[perm]
		newrow = Float64[]
		newval = Float64[]
		oldr = 0
		v = 0
		cnt = 0
		for r in minirow
			cnt += 1
			if r == oldr || oldr == 0
				v += minival[cnt]
			else
				push!(newrow, oldr)
				push!(newval, v)
				v = minival[cnt]
			end
			oldr = r
		end
		push!(newrow, oldr)
		push!(newval, v)

		push!(col, size(newrow, 1))
		append!(row, newrow)
		append!(val, newval)
		# end

		if (preind % (len / 10)) == 1 || preind == len
			num = res.colptr[ncol]
			for c in col
				ncol += 1
				num += c
				res.colptr[ncol] = num
			end
			append!(res.rowval, row)
			append!(res.nzval, val)
			col=Int64[]
			row=Int64[]
			val=Float64[]
		end
		append!(res.rowval, row)
		append!(res.nzval, val)
		row=Int64[]
		val=Float64[]
	end
	# CUDA.CUSOLVER.csreigvsi seems to require CSR not CSC
	return CuSparseMatrixCSR(res)
end

function eigs_ArnoldiMethod(H)
	decomp,history = ArnoldiMethod.partialschur(H,nev=nev,which=ArnoldiMethod.SR())
	e,v = ArnoldiMethod.partialeigen(decomp)
	return e,v
end

function eigs_KrylovKit(H)
	val,vecs,info = KrylovKit.eigsolve(H,rand(eltype(H),size(H,1)),nev,:SR;issymmetric=true)
	return val,vecs
end

println("build H...")
@time H=buildH(diag_,flag_)
println()

println("computing eigenvalues...")
println()

# println("using Arpack:")
# @time e,v = Arpack.eigs(H,nev=nev,which=:SR)
# println()
#
# println("using ArnoldiMethod:")
# @time e,v = eigs_ArnoldiMethod(H)
# println(sort(e))
# println()

# mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)

# println("using KrylovKit:")
# @time e,v = CUDA.@sync eigs_KrylovKit(H)
# println(sort(e))
# println()

println("using CUSOLVER.csreigvsi:")
@time e,v = CUDA.@sync CUDA.CUSOLVER.csreigvsi(H, rand(Float64), CUDA.rand(Float64,len), 1e-6, Cint(1000), 'O')
println(e)
println()


# #=
#
# Some new stuff in preparation for zipper.
#
# =#
#
# #=
# A fusionFlag retains the information of whether main flag is 0.
# =#
# function setFusionFlag!(flag::Vector{Bool},ind::Int64,L::Int64=L+2)
# 	state=ind-1
# 	below=(state>>(2*(L+1)))
# 	start=(state>>(2*L))&3
# 	state=state&(4^L-1)
# 	if (start==3) || (below>=L)
# 		flag[ind] = true
# 		return
# 	end
# 	if state==0 && isodd(L)
# 		flag[ind] = true
# 		return
# 	end
# 	evenxs=iseven(trailingXs(state,L))
# 	tot=start
# 	if(state==1 && iseven(L))
# 		state=0
# 		evenxs=false
# 	end
# 	for pos = 0 : L-1
# 		tot%=3
# 		a=(state >> (2*pos)) & 3
# 		if a==0
# 			evenxs = ! evenxs
# 			if ((pos+1)==below || (below!=0 && pos==L-1))
# 				tot = 3-tot
# 			end
# 		else
# 			if(!evenxs)
# 				flag[ind] = true
# 				return
# 			end
# 			if a==2
# 				tot+=1
# 			elseif a==3
# 				tot+=2
# 			end
# 		end
# 	end
# 	tot=tot-start
# 	if tot<0
# 		tot+=3
# 	end
# 	tot%=3
# 	if state==0 && isodd(L)
# 		flag[ind] = true
# 		return
# 	end
# 	if tot!=0
# 		flag[ind] = true
# 	end
# end
#
# println("preparing fusion flags...")
# println("original...")
# fusionFlag_ = zeros(Bool,4^L)
# @time Threads.@threads for i = 1 : 4^L
# 	setFusionFlag!(fusionFlag_,i,L)
# end
# println()
#
# println("preparing extended fusion flags...")
# println("original...")
# extendedFusionFlag_ = zeros(Bool,4^(L+3)*(L+2))
# @time Threads.@threads for i = 1 : 4^(L+3)*(L+2)
# 	setFusionFlag!(extendedFusionFlag_,i,L+2)
# end
# println()
#
# #=
# Distinguished from Yuji's mainFlag by variable type.
# mainFlag(flag, ...) and mainFlag(fusionFlag, ...) return the same thing.
# =#
# function mainFlag(flag::Vector{Bool},ind::Int64,L::Int64=L)::Bool
# 	return flag[ind]
# end
#
# #=
# The zipper needs to know the edge label (1,a,b,ρ,aρ,a^2ρ) = (1,2,3,4,5,6)
# right before the vertex from which ρ is draped below.
# =#
# function setEdgeAtDrapeMapping!(edgeAtDrapeMapping::Vector{Int8},ind::Int64,L::Int64=L+2)
# 	below = ((ind-1)>>(2*(L+1)))
# 	if below==0
# 		return
# 	end
# 	start = ((ind-1)>>(2*L)) & 3
# 	state = (ind-1)&(4^L-1)
# 	evenxs=iseven(trailingXs(state,L))
# 	if(state==1 && iseven(L))
# 		state=0
# 		evenxs=false
# 	end
# 	tot=start
# 	for pos = 0 : below-2
# 		a=(state >> (2*pos)) & 3
# 		if a==0
# 			evenxs = ! evenxs
# 		elseif a==2
# 			tot+=1
# 		elseif a==3
# 			tot+=2
# 		end
# 	end
# 	tot%=3
# 	if evenxs
# 		edgeAtDrapeMapping[ind] = 4+tot
# 	else
# 		edgeAtDrapeMapping[ind] = 1+tot
# 	end
# end
#
# println("preparing edge at drape mapping...")
# edgeAtDrapeMapping_ = zeros(Int8,4^(L+3)*(L+2))
# @time Threads.@threads for i = 1 : 4^(L+3)*(L+2)
# 	setEdgeAtDrapeMapping!(edgeAtDrapeMapping_,i,L+2)
# end
# println()
#
#
# #=
#
# F-symbol stuff.
#
# =#
#
# const fSymbolMapping_ = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878]
#
# function FSymbolZipper(e1::Int64,s1::Int64,s2::Int64,s3::Int64,s4::Int64)
# 	i = s4 + (s3<<2) + (s2<<4) + (s1<<6) + ((e1-1)<<8)
# 	return fSymbolMapping_[i+1]
# end
#
#
# #=
#
# Zipper stuff.
#
# =#
#
# # Precompute fusion space basis
# println("precompute fusion space bases for zipper...")
# zipBases = fill([], L+1)
# zipLen = fill(0, L+1)
# zipFromInd = fill(Dict(), L+1)
# Threads.@threads for i = 1 : L+1
# 	println(i,"/",L+1)
# 	@time zipBases[i] = filter(x -> (mainFlag(extendedFusionFlag_,x,L+2)==0), 4^(L+3)*i+1 : 4^(L+3)*i+3*4^(L+2))
# 	zipLen[i] = length(zipBases[i])
# 	zipFromInd[i] = Dict((zipBases[i][x],x) for x in 1 : zipLen[i])
# end
# println()
#
# function attachInd(ind::Int64,sp::Tuple{Int64,Int64},start::Int64,L::Int64=L+2)
# 	if ind==2 && iseven(L)
# 		state = 0
# 	else
# 		state = (ind-1)
# 	end
# 	state = state << 2
# 	(a,b)=sp
# 	i = L
# 	state &= ~(3<<(2*(i-1)))
# 	state |= (a<<(2*(i-1)))
# 	j = 1
# 	state &= ~(3<<(2*(j-1)))
# 	state |= (b<<(2*(j-1)))
# 	if (state==0) && iseven(L) && (ind==1)
# 		state = 1
# 	end
# 	return 1+(state+(start<<(2*L))+(1<<(2*(L+1))))
# end
#
# attachPreind(ind::Int64,sp::Tuple{Int64,Int64},start::Int64,L::Int64=L+2) = zipFromInd[1][attachInd(ind,sp,start,L)]
#
# function buildAttach()
# 	col=Int64[]
# 	row=Int64[]
# 	val=Float64[]
# 	for preind = 1 : len
# 		ind = basis[preind]
# 		state = stateFromInd(ind)
# 		if (isodd(trailingXs(state)) || (iseven(L) && ind==2)) # start label is 1
# 			ni = attachPreind(ind,sXX,0,L+2)
# 			append!(col,[preind])
# 			append!(row,[ni])
# 			append!(val,[1])
# 		else
# 			ni = attachPreind(ind,sXX,0,L+2)
# 			append!(col,[preind])
# 			append!(row,[ni])
# 			append!(val,[1/ζ])
# 			ni = attachPreind(ind,s00,0,L+2)
# 			append!(col,[preind])
# 			append!(row,[ni])
# 			append!(val,[ξ])
# 			ni = attachPreind(ind,sPM,1,L+2)
# 			append!(col,[preind])
# 			append!(row,[ni])
# 			append!(val,[ξ])
# 			ni = attachPreind(ind,sMP,2,L+2)
# 			append!(col,[preind])
# 			append!(row,[ni])
# 			append!(val,[ξ])
# 		end
# 	end
# 	append!(col,[len])
# 	append!(row,[zipLen[1]])
# 	append!(val,[0])
# 	return sparse(row,col,val)
# end
#
# function ZipInd(ind::Int64,sp::Tuple{Int64,Int64},L::Int64=L+2)
# 	below = ((ind-1)>>(2*(L+1)))
# 	if below == 0
# 		error("no ρ from below")
# 	end
# 	start = ((ind-1)>>(2*L)) & 3
# 	i = below
# 	below += 1
# 	state = (ind-1) & (4^L-1)
# 	if state==1 && iseven(L)
# 		state = 0
# 	end
# 	(a,b)=sp
# 	state &= ~(3<<(2*(i-1)))
# 	state |= (a<<(2*(i-1)))
# 	j=i+1
# 	state &= ~(3<<(2*(j-1)))
# 	state |= (b<<(2*(j-1)))
# 	if (state==0) && (isodd(trailingXs((ind-1)&(4^L-1),L)) || (iseven(L) && ((ind-1)&(4^L-1))==1))
# 		state = 1
# 	end
# 	return 1+(state+(start<<(2*L))+(below<<(2*(L+1))))
# end
#
# ZipPreind(i::Int64,ind::Int64,sp::Tuple{Int64,Int64},L::Int64=L+2) = zipFromInd[i][ZipInd(ind,sp,L)]
#
# function buildZip(i::Int64)
# 	col=Int64[]
# 	row=Int64[]
# 	val=Float64[]
# 	for preind = 1 : zipLen[i]
# 		ind = zipBases[i][preind]
# 		e1 = Int64(edgeAtDrapeMapping_[ind])
# 		state = stateFromInd(ind,L+2)
# 		s1,s2 = localStatePair(state,i,L+2)
# 		for s3 = 0 : 3
# 			for s4 = 0 : 3
# 				if FSymbolZipper(e1,s1,s2,s3,s4)==0
# 					continue
# 				end
# 				ni = ZipPreind(i+1,ind,(s3,s4),L+2)
# 				append!(col,[preind])
# 				append!(row,[ni])
# 				append!(val,[FSymbolZipper(e1,s1,s2,s3,s4)])
# 			end
# 		end
# 	end
# 	append!(col,[zipLen[i]])
# 	append!(row,[zipLen[i+1]])
# 	append!(val,[0])
# 	return sparse(row,col,val)
# end
#
# function buildDetach()
# 	col=Int64[]
# 	row=Int64[]
# 	val=Float64[]
# 	for preind = 1 : zipLen[L]
# 		ind = zipBases[L+1][preind]
# 		state = stateFromInd(ind,L+2)
# 		ni = 1+(state&(4^L-1))
# 		if ni==1 && ((ind-1)&(4^(L+2)-1))==1 && iseven(L)
# 			ni=2
# 		end
# 		# TODO
# 		if mainFlag(fusionFlag_,ni,L) != 0
# 			continue
# 		end
# 		ni = fromInd[ni]
# 		sp = localStatePair(state,L+1,L+2)
# 		if sp in [sXX, s00, sPM, sMP]
# 			if (isodd(trailingXs(state,L+2)) || iseven(L) && ni==2) # start label is 1
# 				if sp==sXX
# 					append!(col,[preind])
# 					append!(row,[ni])
# 					append!(val,[ζ])
# 				end
# 			else
# 				if sp==sXX
# 					append!(col,[preind])
# 					append!(row,[ni])
# 					append!(val,[1])
# 				else
# 					append!(col,[preind])
# 					append!(row,[ni])
# 					append!(val,[√ζ])
# 				end
# 			end
# 		end
# 	end
# 	append!(col,[zipLen[L]])
# 	append!(row,[len])
# 	append!(val,[0])
# 	return sparse(row,col,val)
# end
#
# println("creating zipper by composition...")
# println("build attach...")
# @time ρ = LinearMap(buildAttach())
#
# zips = fill(ρ, L+1)
# for i = 1 : L
# 	println("build zip ", i, "...")
# 	@time global zips[i] = LinearMap(buildZip(i))
# end
# println("multiply zip...")
# @time for i = 1 : L
# 	global ρ = zips[i] * ρ
# end
#
# println("build detach...")
# @time ρ = LinearMap(buildDetach()) * ρ
# println()
#
#
# #=
#
# Translation (lattice shift) stuff.
#
# =#
#
# function Tind(ind::Int64,L::Int64,right::Bool)
# 	if ind==2 && iseven(L)
# 		return 1
# 	end
# 	if ind==1 && iseven(L)
# 		return 2
# 	end
# 	state = (ind-1) & (4^L-1)
# 	if right
# 		return 1+(state>>2)+((state&3)<<(2*(L-1)))
# 	else
# 		return 1+(state<<2)&(4^L-1)+(state>>(2*(L-1)))
# 	end
# end
#
# function Tfunc!(C::Vector{Float64},B::Vector{Float64},L::Int64=L,right::Bool=true)
# 	Threads.@threads for preind = 1 : len
# 		ind = basis[preind]
# 		C[preind] = B[fromInd[Tind(ind,L,right)]]
# 	end
# end
#
# T=LinearMap((C,B)->Tfunc!(C,B),len,ismutating=true,issymmetric=false,isposdef=false)
#
#
# #=
#
# Simultaneous diagonalization and output.
#
# =#
#
# # Print as mathematica array to reuse Mathematica code for making plots.
# function mathematicaVector(V::Vector{Float64})
# 	s="{"
# 	for i = 1:(size(V,1)-1)
# 		s*=string(V[i])
# 		s*=", "
# 	end
# 	s*=string(V[size(V,1)])
# 	s*="}"
# 	return s
# end
#
# # Print as mathematica matrix to reuse Mathematica code for making plots.
# function mathematicaMatrix(M::Vector{Vector{Float64}})
# 	s="{\n"
# 	for i = 1:(size(M,1)-1)
# 		s*=mathematicaVector(M[i])
# 		s*=",\n"
# 	end
# 	s*=mathematicaVector(M[size(M,1)])
# 	s*="\n}\n"
# 	return s
# end
#
# function diagonalizeHTρ(e,v,T,ρ)
# 	println("diagonalizing H,T,ρ...")
#
# 	smallH = Matrix(diagm(e))
# 	smallT = Matrix(adjoint(v)*T*v)
# 	smallρ = Matrix(adjoint(v)*ρ*v)
# 	smalle,smallv = eigen(smallH+smallT+smallρ)
#
# 	Hs = real(diag(adjoint(smallv)*smallH*smallv))
# 	Ts = complex(diag(adjoint(smallv)*smallT*smallv))
# 	Ps = real(map(log,Ts)/(2*π*im))*L
# 	ρs = real(diag(adjoint(smallv)*smallρ*smallv))
# 	HPρs = hcat(Hs,Ps,ρs)
# 	HPρs = sort([HPρs[i,:] for i in 1:size(HPρs, 1)])
# 	s=""
# 	s*=string(HPρs[1][1])
# 	print(mathematicaMatrix(HPρs))
# end
#
# @time diagonalizeHTρ(e,v,T,ρ)
