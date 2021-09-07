using LinearAlgebra,LinearMaps
using SparseArrays
using Arpack
using ArnoldiMethod
using KrylovKit
using BenchmarkTools
using JLD2

const L = 18
const nev = 1
const dataPath = "data/"

# const L = parse(Int64, ARGS[1])
# const nev = parse(Int64, ARGS[2])
# const dataPath = "/n/holyscratch01/yin_lab/Users/yhlin/ed/" # NOTE If on cluster set to scratch space

const eigSolver = "Arpack" # "Arpack" "ArnoldiMethod" "KrylovKit"
const onlyT = true # compute eigenstates of H and measure T but not ρ
const buildSparse = true # use sparse matrices and not LinearMap


#=
Save/load hard disk to reduce memory usage when measuring ρ.
=#

const dataPathL = dataPath * string(L) * "/"
if !ispath(dataPath)
	mkdir(dataPath)
end
if !ispath(dataPathL)
	mkdir(dataPathL)
end
if !ispath(dataPathL * "rhov/")
	mkdir(dataPathL * "rhov/")
end
if !ispath(dataPathL * "prep/")
	mkdir(dataPathL * "prep/")
end
if !ispath(dataPathL * "edge/")
	mkdir(dataPathL * "edge/")
end
if !ispath(dataPathL * "zip/")
	mkdir(dataPathL * "zip/")
end
if !ispath(dataPathL * "done/")
	mkdir(dataPathL * "done/")
end

println()
flush(stdout)
println("exact diagonalization of L=", L, " with build sparse=", buildSparse, " and keeping nev=", nev)
println()
flush(stdout)

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
Divide-and-conquer to construct basis.
=#

# key is (L, evenxs_left, evenxs_right, Z3 charge)
# value is state (not ind)
basisLego_ = Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}}()

# initialize divide and conquer
for el = false : true
	for er = false : true
		for q = 0 : 2
			basisLego_[(0, el, er, q)] = []
			basisLego_[(1, el, er, q)] = []
		end
	end
end
basisLego_[(1, true, true, 0)] = [ 1 ]
basisLego_[(1, true, true, 1)] = [ 2 ]
basisLego_[(1, true, true, 2)] = [ 3 ]

function setBasisLego!(basisLego::Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}},
	L::Int64, evenxs_left::Bool, evenxs_right::Bool, q::Int64)
	basisLego[(L, evenxs_left, evenxs_right, q)] = []
	if evenxs_right
		# append X
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], getBasisLego(basisLego, L-1, evenxs_left, false, q) )
		# append non-X
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], [ x + (1 << (2*(L-1))) for x in getBasisLego(basisLego, L-1, evenxs_left, true, q) ] )
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], [ x + (2 << (2*(L-1))) for x in getBasisLego(basisLego, L-1, evenxs_left, true, (q+2)%3) ] )
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], [ x + (3 << (2*(L-1))) for x in getBasisLego(basisLego, L-1, evenxs_left, true, (q+1)%3) ] )
	else
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], getBasisLego(basisLego, L-1, evenxs_left, true, q) )
	end
	if ((iseven(L) && !evenxs_left) || (isodd(L) && evenxs_left)) && evenxs_right
		# append non-X to XX...X
		append!( basisLego[(L, evenxs_left, evenxs_right, q)], [ (q+1) << (2*(L-1)) ] )
	end
end

# basis of states (not inds)
function getBasisLego(basisLego::Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}},
	L::Int64, evenxs_left::Bool, evenxs_right::Bool, q::Int64)::Vector{Int64}
	if !haskey(basisLego, (L, evenxs_left, evenxs_right, q))
		setBasisLego!(basisLego, L, evenxs_left, evenxs_right, q)
	end
	return basisLego[(L, evenxs_left, evenxs_right, q)]
end

# basis of states (not inds)
function getBasisLego(basisLego::Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}}, L::Int64)::Vector{Int64}
	res = []
	append!(res, getBasisLego(basisLego, L, true, true, 0))
	append!(res, getBasisLego(basisLego, L, false, false, 0))
	if iseven(L)
		append!(res, [0, 1])
	end
	sort!(res)
	return res
end

const basisPath = dataPathL * "basis.jld2"
if ispath(basisPath)
	println("load basis...")
	flush(stdout)
	@time @load basisPath basis len fromInd
else
	println("compute basis...")
	flush(stdout)
	@time const basis = [ x+1 for x in getBasisLego(basisLego_, L) ]
	const len = length(basis)
	@time const fromInd = Dict((basis[x],x) for x in 1:len)
	@time @save basisPath basis len fromInd
end
println()
flush(stdout)


#=
Construct Hamiltonian.
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

flag_ = zeros(Int32,len)

function setFlag!(flag::Vector{Int32},preind::Int64,L::Int64=L)
	ind = basis[preind]
	state=ind-1
	below=(state>>(2*(L+1)))
	start=(state>>(2*L))&3
	state=state&(4^L-1)
	if (start==3) || (below>=L)
		return
	end
	if state==0 && isodd(L)
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
		return
	end
end

diag_ = zeros(Float32,len)

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

function isρ1ρ(flag::Vector{Int32},preind::Int64,i::Int64)
	return ((flag[preind] >> (i-1)) & 1) == 1
end

function computeDiag!(diag::Vector{Float32},flag::Vector{Int32},preind::Int64)
	ind = basis[preind]
	state=stateFromInd(ind)
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
flush(stdout)

const prepPath = dataPathL * "prep/prep.jld2"
if ispath(prepPath)
	println("load flag and diag...")
	flush(stdout)
	@time @load prepPath flag_ diag_
else
	println("compute flag and diag...")
	flush(stdout)
	Threads.@threads for preind = 1 : len
		setFlag!(flag_,preind)
		computeDiag!(diag_,flag_,preind)
	end
	@time @save prepPath flag_ diag_
end
println()
flush(stdout)

newPreind(state,i,sp) = fromInd[newInd(state,i,sp)]

TT = Union{Vector{Float32},Vector{Float16}}

function sortAndAppendColumn!(
	col::Vector{Int32},
	row::Vector{Int32},
	val::TT,
	miniRow::Vector{Int64},
	miniVal::TT
	)
	perm = sortperm(miniRow)
	miniRow = miniRow[perm]
	miniVal = miniVal[perm]
	newMiniRow = Int32[]
	newMinival = Float32[]
	oldr = 0
	v = 0
	cnt = 0
	for r in miniRow
		cnt += 1
		if r == oldr || oldr == 0
			v += miniVal[cnt]
		else
			push!(newMiniRow, oldr)
			push!(newMinival, v)
			v = miniVal[cnt]
		end
		oldr = r
	end
	push!(newMiniRow, oldr)
	push!(newMinival, v)
	push!(col, size(newMiniRow, 1))
	append!(row, newMiniRow)
	append!(val, newMinival)
end

function buildH(diag,flag)
	res = sparse(Int32[],Int32[],Float32[],len,len)
	col=Int32[]
	row=Int32[]
	val=Float32[]
	ncol = 1
	for preind = 1 : len
		ind = basis[preind]
		state=stateFromInd(ind)
		# cannot directly append into row, val because row needs to be ordered
		# define miniRow/val and append to row/val after sorting
		miniRow = [preind]
		miniVal = [diag[preind]]
		for i = 1 : L
			sp=localStatePair(state,i)
			if sp==sXX  && isρ1ρ(flag,preind,i)
				append!(miniRow,map(s->newPreind(state,i,s),[sPM,sMP,s00]))
				append!(miniVal,-ξ .* [y1,y2,x])
			elseif sp==sPM
				append!(miniRow,map(s->newPreind(state,i,s),[sXX,sMP,s00]))
				append!(miniVal,-y1 .* [ξ,y2,x])
			elseif sp==sMP
				append!(miniRow,map(s->newPreind(state,i,s),[sXX,sPM,s00]))
				append!(miniVal,-y2 .* [ξ,y1,x])
			elseif sp==s00
				append!(miniRow,map(s->newPreind(state,i,s),[sXX,sPM,sMP]))
				append!(miniVal,-x .* [ξ,y1,y2])
			elseif sp==s0P
				append!(miniRow,map(s->newPreind(state,i,s),[sP0,sMM]))
				append!(miniVal,-y1 .* [y2,z])
			elseif sp==sP0
				append!(miniRow,map(s->newPreind(state,i,s),[s0P,sMM]))
				append!(miniVal,-y2 .* [y1,z])
			elseif sp==sMM
				append!(miniRow,map(s->newPreind(state,i,s),[s0P,sP0]))
				append!(miniVal,-z .* [y1,y2])
			elseif sp==s0M
				append!(miniRow,map(s->newPreind(state,i,s),[sM0,sPP]))
				append!(miniVal,-y2 .* [y1,z])
			elseif sp==sM0
				append!(miniRow,map(s->newPreind(state,i,s),[s0M,sPP]))
				append!(miniVal,-y1 .* [y2,z])
			elseif sp==sPP
				append!(miniRow,map(s->newPreind(state,i,s),[s0M,sM0]))
				append!(miniVal,-z .* [y2,y1])
			end
		end

		sortAndAppendColumn!(col, row, val, miniRow, miniVal)

		if (preind % (len / 10)) == 1 || preind == len
			num = res.colptr[ncol]
			for c in col
				ncol += 1
				num += c
				res.colptr[ncol] = num
			end
			append!(res.rowval, row)
			append!(res.nzval, val)
			col=Int32[]
			row=Int32[]
			val=Float32[]
		end
		append!(res.rowval, row)
		append!(res.nzval, val)
		row=Int32[]
		val=Float32[]
	end
	return res
end

function Hfunc!(C,B,diag::Vector{Float32},flag::Vector{Int32})
	Threads.@threads for preind = 1 : len
		C[preind] = diag[preind] * B[preind]
	end
	for i = 1 : L
		Threads.@threads for preind = 1 : len
			ind = basis[preind]
			state=stateFromInd(ind)
			sp=localStatePair(state,i)
			if sp==sXX  && isρ1ρ(flag,preind,i)
				C[preind]-= ξ * (
				     y1 * B[newPreind(state,i,sPM)] +
					 y2 * B[newPreind(state,i,sMP)] +
					 x  * B[newPreind(state,i,s00)]
			    )
			elseif sp==sPM
				C[preind]-= y1*(
					 ξ * B[newPreind(state,i,sXX)] +
					 y2* B[newPreind(state,i,sMP)] +
					 x * B[newPreind(state,i,s00)]
			    )
			elseif sp==sMP
				C[preind]-= y2*(
					 ξ * B[newPreind(state,i,sXX)] +
					 y1* B[newPreind(state,i,sPM)] +
					 x * B[newPreind(state,i,s00)]
			    )
			elseif sp==s00
				C[preind]-= x*(
					 ξ * B[newPreind(state,i,sXX)] +
					 y1* B[newPreind(state,i,sPM)] +
					 y2* B[newPreind(state,i,sMP)]
			    )
			elseif sp==s0P
				C[preind]-= y1*(
					y2 * B[newPreind(state,i,sP0)] +
					z  * B[newPreind(state,i,sMM)]
				)
			elseif sp==sP0
				C[preind]-= y2*(
					y1 * B[newPreind(state,i,s0P)] +
					z  * B[newPreind(state,i,sMM)]
				)
			elseif sp==sMM
				C[preind]-= z*(
					y1 * B[newPreind(state,i,s0P)] +
					y2 * B[newPreind(state,i,sP0)]
				)
			elseif sp==s0M
				C[preind]-= y2*(
					y1 * B[newPreind(state,i,sM0)] +
					z  * B[newPreind(state,i,sPP)]
				)
			elseif sp==sM0
				C[preind]-= y1*(
					y2 * B[newPreind(state,i,s0M)] +
					z  * B[newPreind(state,i,sPP)]
				)
			elseif sp==sPP
				C[preind]-= z*(
					y2 * B[newPreind(state,i,s0M)] +
					y1 * B[newPreind(state,i,sM0)]
				)
			end
		end
	end
end


#=
Diagonalize Hamiltonian.
=#

function eigs_ArnoldiMethod(H)
	decomp,history = ArnoldiMethod.partialschur(H,nev=nev,which=ArnoldiMethod.SR())
	e,v = ArnoldiMethod.partialeigen(decomp)
	return e,v
end

function eigs_KrylovKit(H)
	val,vecs,info = KrylovKit.eigsolve(H,rand(eltype(H),size(H,1)),nev,:SR;issymmetric=true, krylovdim=2*nev)
	mat = zeros(size(vecs,1),len)
	mat = zeros(len,size(vecs,1))
	cnt = 1
	for v in vecs
		mat[:,cnt] = v
		cnt += 1
	end
	return val,mat
end

const eigPath = dataPathL * "eig.jld2"
if ispath(eigPath)
	println("load eigen...")
	flush(stdout)
	@time @load eigPath e v
end

if !ispath(eigPath) || length(e) < nev
	if buildSparse
		HPath = dataPathL * "H.jld2"
		if ispath(HPath)
			println("load H...")
			flush(stdout)
			@load HPath H
		else
			println("build H...")
			flush(stdout)
			@time H=buildH(diag_,flag_)
			@save HPath H
		end
		println()
		flush(stdout)
	else
		H=LinearMap((C,B)->Hfunc!(C,B,diag_,flag_),len,ismutating=true,issymmetric=true,isposdef=false)
	end

	println("compute eigen...")
	println()
	flush(stdout)

	if eigSolver == "Arpack"
		println("using Arpack:")
		flush(stdout)
		@time e,v = Arpack.eigs(H,nev=nev,which=:SR)
		println(sort(real(e)))
	elseif eigSolver == "ArnoldiMethod"
		println("using ArnoldiMethod:")
		flush(stdout)
		@time e,v = eigs_ArnoldiMethod(H)
		println(sort(e))
	elseif eigSolver == "KrylovKit"
		println("using KrylovKit:")
		flush(stdout)
		@time e,v = eigs_KrylovKit(H)
		println(sort(e))
		@time @save eigPath e v
	else
		println("invalid eigensolver...bye")
		flush(stdout)
		exit()
	end
end
if length(e) > nev
	e = e[1:nev]
	v = v[:,1:nev]
end
println()
flush(stdout)


#=
Divide and conquer for extended chain.
=#

function extendedInd(
	state1::Int64,
	state2::Int64,
	L1::Int64,
	L2::Int64,
	start::Int64,
	below::Int64,
	s1::Int64,
	s2::Int64)

	return 1 + state1 + (s1 << (2*L1)) + (state2 << (2*(L1+1))) + (s2 << (2*(L1+L2+1))) + (start << (2*(L1+L2+2))) + (below << (2*(L1+L2+3)))
end

# basis of inds (not states)
function getExtendedBasis(
	basisLego::Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}},
	L::Int64,
	start::Int64,
	below::Int64
	)::Vector{Int64}

	L1 = below-1
	L2 = L-L1
	res = []

	### L1 L2 both not all X ###
	for q1 = 0 : 2
		for q2 = 0 : 2
			# below is s1 = s2 = X
			tot = - (q2 - (q1 + start))
			if mod(tot, 3) == start
				for evenxs_left = false : true
					for evenxs_right = false : true
						for state1 in getBasisLego(basisLego, L1, evenxs_left, evenxs_right, q1)
							for state2 in getBasisLego(basisLego, L2, !evenxs_right, !evenxs_left, q2)
								append!(res, extendedInd( state1, state2, L1, L2, start, below, 0, 0 ))
							end
						end
					end
				end
			end
			# below is s1 = X, s2 ≂̸ X
			for s2 = 1 : 3
				tot = (s2 - 1) + q2 - (q1 + start)
				if mod(tot, 3) == start
					evenxs_left = true
					for evenxs_right = false : true
						for state1 in getBasisLego(basisLego, L1, evenxs_left, evenxs_right, q1)
							for state2 in getBasisLego(basisLego, L2, !evenxs_right, evenxs_left, q2)
								append!(res, extendedInd( state1, state2, L1, L2, start, below, 0, s2 ))
							end
						end
					end
				end
			end
			# below is s1 ≂̸ X, s2 = X
			for s1 = 1 : 3
				tot = - (q2 + (s1 - 1) + (q1 + start))
				if mod(tot, 3) == start
					evenxs_right = true
					for evenxs_left = false : true
						for state1 in getBasisLego(basisLego, L1, evenxs_left, evenxs_right, q1)
							for state2 in getBasisLego(basisLego, L2, evenxs_right, !evenxs_left, q2)
								append!(res, extendedInd( state1, state2, L1, L2, start, below, s1, 0 ))
							end
						end
					end
				end
			end
			# below is s1 ≂̸ X, s2 ≂̸ X
			for s1 = 1 : 3
				for s2 = 1 : 3
					tot = (s2 - 1) + (q2 + (s1 - 1) + (q1 + start))
					if mod(tot, 3) == start
						evenxs_left = true
						evenxs_right = true
						for state1 in getBasisLego(basisLego, L1, evenxs_left, evenxs_right, q1)
							for state2 in getBasisLego(basisLego, L2, evenxs_right, evenxs_left, q2)
								append!(res, extendedInd( state1, state2, L1, L2, start, below, s1, s2 ))
							end
						end
					end
				end
			end
		end
	end

	### L1 is all X, L2 is not ###
	q1 = 0
	for q2 = 0 : 2
		# below is s1 = s2 = X
		tot = - (q2 - (q1 + start))
		if mod(tot, 3) == start
			if iseven(L1)
				for state2 in getBasisLego(basisLego, L2, true, true, q2)
					append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, 0 ))
				end
				for state2 in getBasisLego(basisLego, L2, false, false, q2)
					append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, 0 ))
				end
			else
				for state2 in getBasisLego(basisLego, L2, true, false, q2)
					append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, 0 ))
				end
				for state2 in getBasisLego(basisLego, L2, false, true, q2)
					append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, 0 ))
				end
			end
		end
		# below is s1 = X, s2 ≂̸ X
		for s2 = 1 : 3
			tot = (s2 - 1) + q2 - (q1 + start)
			if mod(tot, 3) == start
				if iseven(L1)
					for state2 in getBasisLego(basisLego, L2, false, true, q2)
						append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, s2 ))
					end
				else
					for state2 in getBasisLego(basisLego, L2, true, true, q2)
						append!(res, extendedInd( 0, state2, L1, L2, start, below, 0, s2 ))
					end
				end
			end
		end
		# below is s1 ≂̸ X, s2 = X
		for s1 = 1 : 3
			tot = - (q2 + (s1 - 1) + (q1 + start))
			if mod(tot, 3) == start
				if iseven(L1)
					for state2 in getBasisLego(basisLego, L2, true, false, q2)
						append!(res, extendedInd( 0, state2, L1, L2, start, below, s1, 0 ))
					end
				else
					for state2 in getBasisLego(basisLego, L2, true, true, q2)
						append!(res, extendedInd( 0, state2, L1, L2, start, below, s1, 0 ))
					end
				end
			end
		end
		# below is s1 ≂̸ X, s2 ≂̸ X
		for s1 = 1 : 3
			for s2 = 1 : 3
				tot = (s2 - 1) + (q2 + (s1 - 1) + (q1 + start))
				if mod(tot, 3) == start && iseven(L1)
					for state2 in getBasisLego(basisLego, L2, true, true, q2)
						append!(res, extendedInd( 0, state2, L1, L2, start, below, s1, s2 ))
					end
				end
			end
		end
	end

	### L2 is all X, L1 is not ###
	q2 = 0
	for q1 = 0 : 2
		# below is s1 = s2 = X
		tot = - (q2 - (q1 + start))
		if mod(tot, 3) == start
			if iseven(L2)
				for state1 in getBasisLego(basisLego, L1, true, true, q1)
					append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, 0 ))
				end
				for state1 in getBasisLego(basisLego, L1, false, false, q1)
					append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, 0 ))
				end
			else
				for state1 in getBasisLego(basisLego, L1, true, false, q1)
					append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, 0 ))
				end
				for state1 in getBasisLego(basisLego, L1, false, true, q1)
					append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, 0 ))
				end
			end
		end
		# below is s1 = X, s2 ≂̸ X
		for s2 = 1 : 3
			tot = (s2 - 1) + q2 - (q1 + start)
			if mod(tot, 3) == start
				if iseven(L2)
					for state1 in getBasisLego(basisLego, L1, true, false, q1)
						append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, s2 ))
					end
				else
					for state1 in getBasisLego(basisLego, L1, true, true, q1)
						append!(res, extendedInd( state1, 0, L1, L2, start, below, 0, s2 ))
					end
				end
			end
		end
		# below is s1 ≂̸ X, s2 = X
		for s1 = 1 : 3
			tot = - (q2 + (s1 - 1) + (q1 + start))
			if mod(tot, 3) == start
				if iseven(L2)
					for state1 in getBasisLego(basisLego, L1, false, true, q1)
						append!(res, extendedInd( state1, 0, L1, L2, start, below, s1, 0 ))
					end
				else
					for state1 in getBasisLego(basisLego, L1, true, true, q1)
						append!(res, extendedInd( state1, 0, L1, L2, start, below, s1, 0 ))
					end
				end
			end
		end
		# below is s1 ≂̸ X, s2 ≂̸ X
		for s1 = 1 : 3
			for s2 = 1 : 3
				tot = (s2 - 1) + (q2 + (s1 - 1) + (q1 + start))
				if mod(tot, 3) == start && iseven(L2)
					for state1 in getBasisLego(basisLego, L1, true, true, q1)
						append!(res, extendedInd( state1, 0, L1, L2, start, below, s1, s2 ))
					end
				end
			end
		end
	end

	### both L1 and L2 are all X ###
	if iseven(L1) && iseven(L2)
		append!(res, extendedInd( 0, 0, L1, L2, start, below, 0, 0 ))
		for s1 = 1 : 3
			s2 = 1 + mod(1-s1,3)
			append!(res, extendedInd( 0, 0, L1, L2, start, below, s1, s2 ))
		end
	end
	if isodd(L1) && isodd(L2)
		append!(res, extendedInd( 0, 0, L1, L2, start, below, 0, 0 ))
	end
	if (iseven(L1) && isodd(L2) || iseven(L2) && isodd(L1))
		for s2 = 1 : 3
			tot = (s2 - 1) - start
			if mod(tot,3) == start
				append!(res, extendedInd( 0, 0, L1, L2, start, below, 0, s2 ))
			end
		end
		for s1 = 1 : 3
			tot = - ( (s1 - 1) + start )
			if mod(tot,3) == start
				append!(res, extendedInd( 0, 0, L1, L2, start, below, s1, 0 ))
			end
		end
	end

	sort!(res)
	return res
end

# basis of inds (not states)
function getExtendedBasis(
	basisLego::Dict{Tuple{Int8,Bool,Bool,Int8},Vector{Int64}},
	L::Int64,
	below::Int64
	)::Vector{Int64}

	res = []
	for start = 0 : 2
		append!(res, getExtendedBasis(basisLego, L, start, below))
		if iseven(L)
			append!(res, 1 + 1 + (start << (2*(L+2))) + (below << (2*(L+3))))
		end
	end
	return res
end


#=
F-symbol stuff.
=#

const fSymbolMapping_ = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.3027756377319946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5502505227003375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5351837584879964, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5542656654188897, 0.0, 0.0, 0.7675918792439983, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3218575446628878]

function FSymbolZipper(e1::Int64,s1::Int64,s2::Int64,s3::Int64,s4::Int64)
	i = s4 + (s3<<2) + (s2<<4) + (s1<<6) + ((e1-1)<<8)
	return fSymbolMapping_[i+1]
end


#=
Construct zipper.
=#

function attachInd(ind::Int64,sp::Tuple{Int64,Int64},start::Int64,L::Int64=L+2)
	if ind==2 && iseven(L)
		state = 0
	else
		state = (ind-1)
	end
	state = state << 2
	(a,b)=sp
	i = L
	state &= ~(3<<(2*(i-1)))
	state |= (a<<(2*(i-1)))
	j = 1
	state &= ~(3<<(2*(j-1)))
	state |= (b<<(2*(j-1)))
	if (state==0) && iseven(L) && (ind==1)
		state = 1
	end
	return 1+(state+(start<<(2*L))+(1<<(2*(L+1))))
end

attachPreind(ind::Int64,sp::Tuple{Int64,Int64},start::Int64,L::Int64=L+2) = zipFromInd[attachInd(ind,sp,start,L)]

function attach!(C,B)
	Threads.@threads for preind = 1 : ziplen
		C[preind] = 0
	end
	for preind = 1 : len
		ind = basis[preind]
		if B[preind] == 0
			continue
		end
		state = stateFromInd(ind)
		if (isodd(trailingXs(state)) || (iseven(L) && ind==2))
			ni = attachPreind(ind,sXX,0,L+2)
			C[ni] += B[preind]
		else
			ni = attachPreind(ind,sXX,0,L+2)
			C[ni] += 1/ζ * B[preind]
			ni = attachPreind(ind,s00,0,L+2)
			C[ni] += ξ * B[preind]
			ni = attachPreind(ind,sPM,1,L+2)
			C[ni] += ξ * B[preind]
			ni = attachPreind(ind,sMP,2,L+2)
			C[ni] += ξ * B[preind]
		end
	end
end

function buildAttach()
	col=Int32[]
	row=Int32[]
	val=Float16[]
	for preind = 1 : len
		ind = basis[preind]
		state = stateFromInd(ind)
		if (isodd(trailingXs(state)) || (iseven(L) && ind==2))
			ni = attachPreind(ind,sXX,0,L+2)
			append!(col,[preind])
			append!(row,[ni])
			append!(val,[1])
		else
			ni = attachPreind(ind,sXX,0,L+2)
			append!(col,[preind])
			append!(row,[ni])
			append!(val,[1/ζ])
			ni = attachPreind(ind,s00,0,L+2)
			append!(col,[preind])
			append!(row,[ni])
			append!(val,[ξ])
			ni = attachPreind(ind,sPM,1,L+2)
			append!(col,[preind])
			append!(row,[ni])
			append!(val,[ξ])
			ni = attachPreind(ind,sMP,2,L+2)
			append!(col,[preind])
			append!(row,[ni])
			append!(val,[ξ])
		end
	end
	append!(col,[len])
	append!(row,[ziplen])
	append!(val,[0])
	return sparse(row,col,val)
end

function zipInd(ind::Int64,sp::Tuple{Int64,Int64},L::Int64=L+2)
	below = ((ind-1)>>(2*(L+1)))
	if below == 0
		error("no ρ from below")
	end
	start = ((ind-1)>>(2*L)) & 3
	i = below
	below += 1
	state = (ind-1) & (4^L-1)
	if state==1 && iseven(L)
		state = 0
	end
	(a,b)=sp
	state &= ~(3<<(2*(i-1)))
	state |= (a<<(2*(i-1)))
	j=i+1
	state &= ~(3<<(2*(j-1)))
	state |= (b<<(2*(j-1)))
	if (state==0) && (isodd(trailingXs((ind-1)&(4^L-1),L)) || (iseven(L) && ((ind-1)&(4^L-1))==1))
		state = 1
	end
	return 1+(state+(start<<(2*L))+(below<<(2*(L+1))))
end

zipPreInd(i::Int64,ind::Int64,sp::Tuple{Int64,Int64},L::Int64=L+2) = zipFromInd[zipInd(ind,sp,L)]

function zip!(C,B,i::Int64)
	Threads.@threads for preind = 1 : ziplen
		C[preind] = 0
	end
	for preind = 1 : ziplen
		ind = inBasis[preind]
		if B[preind] == 0
			continue
		end
		e1 = Int64(edgeAtDrapeMapping[preind])
		state = stateFromInd(ind,L+2)
		s1,s2 = localStatePair(state,i,L+2)
		for s3 = 0 : 3
			for s4 = 0 : 3
				if FSymbolZipper(e1,s1,s2,s3,s4)==0
					continue
				end
				ni = zipPreInd(i+1,ind,(s3,s4),L+2)
				C[ni] += FSymbolZipper(e1,s1,s2,s3,s4) * B[preind]
			end
		end
	end
end

function buildZip(i::Int64)
	res = sparse(Int32[],Int32[],Float16[],ziplen,ziplen)
	col=Int32[]
	row=Int32[]
	val=Float16[]
	ncol = 1
	for preind = 1 : ziplen
		miniRow = Int64[]
		miniVal = Float16[]
		ind = inBasis[preind]
		e1 = Int64(edgeAtDrapeMapping[preind])
		state = stateFromInd(ind,L+2)
		s1,s2 = localStatePair(state,i,L+2)
		for s3 = 0 : 3
			for s4 = 0 : 3
				if FSymbolZipper(e1,s1,s2,s3,s4)==0
					continue
				end
				ni = zipPreInd(i+1,ind,(s3,s4),L+2)
				append!(miniRow,[ni])
				append!(miniVal,[FSymbolZipper(e1,s1,s2,s3,s4)])
			end
		end
		sortAndAppendColumn!(col, row, val, miniRow, miniVal)
		if (preind % (ziplen / 10)) == 1 || preind == ziplen
			num = res.colptr[ncol]
			for c in col
				ncol += 1
				num += c
				res.colptr[ncol] = num
			end
			append!(res.rowval, row)
			append!(res.nzval, val)
			col=Int32[]
			row=Int32[]
			val=Float16[]
		end
		append!(res.rowval, row)
		append!(res.nzval, val)
		row=Int32[]
		val=Float16[]
	end
	return res
end

function detach!(C,B)
	Threads.@threads for preind = 1 : len
		C[preind] = 0
	end
	for preind = 1 : ziplen
		ind = inBasis[preind]
		if B[preind] == 0
			continue
		end
		state = stateFromInd(ind,L+2)
		ni = 1+(state&(4^L-1))
		if ni==1 && ((ind-1)&(4^(L+2)-1))==1 && iseven(L)
			ni=2
		end
		if !haskey(fromInd, ni)
			continue
		end
		ni = fromInd[ni]
		sp = localStatePair(state,L+1,L+2)
		if (isodd(trailingXs(state,L+2)) || iseven(L) && ni==2)
			if sp==sXX
				C[ni] += ζ * B[preind]
			end
		else
			if sp==sXX
				C[ni] += B[preind]
			elseif sp==s00 || sp==sPM || sp==sMP
				C[ni] += √ζ * B[preind]
			end
		end
	end
end

function buildDetach()
	col=Int32[]
	row=Int32[]
	val=Float16[]
	for preind = 1 : ziplen
		ind = inBasis[preind]
		state = stateFromInd(ind,L+2)
		ni = 1+(state&(4^L-1))
		if ni==1 && ((ind-1)&(4^(L+2)-1))==1 && iseven(L)
			ni=2
		end
		if !haskey(fromInd, ni)
			continue
		end
		ni = fromInd[ni]
		sp = localStatePair(state,L+1,L+2)
		if (isodd(trailingXs(state,L+2)) || iseven(L) && ni==2)
			if sp==sXX
				append!(col,[preind])
				append!(row,[ni])
				append!(val,[ζ])
			end
		else
			if sp==sXX
				append!(col,[preind])
				append!(row,[ni])
				append!(val,[1])
			elseif sp==s00 || sp==sPM || sp==sMP
				append!(col,[preind])
				append!(row,[ni])
				append!(val,[√ζ])
			end
		end
	end
	append!(col,[ziplen])
	append!(row,[len])
	append!(val,[0])
	return sparse(row,col,val)
end

#=
The zipper needs to know the edge label (1,a,b,ρ,aρ,a^2ρ) = (1,2,3,4,5,6)
right before the vertex from which ρ is draped below.
=#
function setEdgeAtDrapeMapping!(edgeAtDrapeMapping,below::Int64,preind::Int64)
	state = inBasis[preind]-1
	start = (state>>(2*(L+2))) & 3
	state = state&(4^(L+2)-1)
	evenxs=iseven(trailingXs(state,L+2))
	if(state==1 && iseven(L))
		state=0
		evenxs=false
	end
	tot=start
	for pos = 0 : below-2
		a=(state >> (2*pos)) & 3
		if a==0
			evenxs = ! evenxs
		elseif a==2
			tot+=1
		elseif a==3
			tot+=2
		end
	end
	tot%=3
	if evenxs
		edgeAtDrapeMapping[preind] = 4+tot
	else
		edgeAtDrapeMapping[preind] = 1+tot
	end
end


#=
Act zipper.
=#

# Compute extended basis etc to prepare for one step of zipper.
function prepare!(below::Int64)
	preparePath = dataPathL * "prep/prep_" * string(below) * ".jld2"
	if ispath(preparePath)
		if below == 0
			println("load prepare attach...")
		elseif below == L+1
			println("load prepare detach...")
		else
			println("load prepare zip " * string(below) * "...")
		end
		@load preparePath inBasis outBasis zipFromInd edgeAtDrapeMapping
	else
		if below == 0
			println("prepare attach...")
		elseif below == L+1
			println("prepare detach...")
		else
			println("prepare zip " * string(below) * "...")
		end
		if below == 0
			global inBasis = basis
		else
			global inBasis = outBasis
		end
		if below == L+1
			global outBasis = basis
			global zipFromInd = fromInd
		else
			if below > 0
				global outBasis = getExtendedBasis(basisLego_, L, below+1)
			end
			global zipFromInd = Dict((outBasis[x],Int32(x)) for x in 1 : ziplen)
		end

		global edgeAtDrapeMapping
		if 1 <= below <= L
			for preind = 1 : ziplen
				setEdgeAtDrapeMapping!(edgeAtDrapeMapping,below,preind)
			end
		end

		@save preparePath inBasis outBasis zipFromInd edgeAtDrapeMapping
	end
end

if !onlyT
	const prepZipPath = dataPathL * "prepZip.jld2"
	if ispath(prepZipPath)
		println("load zipper...")
		flush(stdout)
		@time @load prepZipPath inBasis outBasis ziplen zipFromInd edgeAtDrapeMapping
	else
		println("prepare zipper...")
		flush(stdout)
		inBasis = basis
		@time outBasis = getExtendedBasis(basisLego_, L, 1)
		ziplen = length(outBasis)
		zipFromInd = Dict{Int64,Int32}((outBasis[x],Int32(x)) for x in 1 : ziplen)
		edgeAtDrapeMapping = zeros(Int8,ziplen)
		@time @save prepZipPath inBasis outBasis ziplen zipFromInd edgeAtDrapeMapping
	end
	println()
	flush(stdout)
end

function ρMatrix(v)
	if buildSparse
		@time prepare!(0)

		donePath = dataPathL * "done/done_" * string(nev) * "_" * string(0) * ".jld2"
		if !ispath(donePath)
			attachPath = dataPathL * "attach.jld2"
			if ispath(attachPath)
				println("load attach...")
				flush(stdout)
				@time @load attachPath ρ
			else
				println("build attach...")
				flush(stdout)
				@time ρ = buildAttach()
				@time @save attachPath ρ
			end

			println("act attach...")
			flush(stdout)
			@time for s = 1 : length(e)
				println(s)
				flush(stdout)
				path = dataPathL * "rhov/rhov_0_" * string(s) * ".jld2"
				if !ispath(path)
					@time rhov = ρ * v[:,s]
					@save path rhov
					flush(stdout)
				end
			end
		end
		@save donePath donePath
		println()
		flush(stdout)

		for i = 1 : L
			@time prepare!(i)
			donePath = dataPathL * "done/done_" * string(nev) * "_" * string(i) * ".jld2"
			if !ispath(donePath)
				zipPath = dataPathL * "zip/zip_" * string(i) * ".jld2"
				if ispath(zipPath)
					println("load zip ", i, "...")
					flush(stdout)
					@time @load zipPath ρ
				else
					println("build zip ", i, "...")
					flush(stdout)
					@time ρ = buildZip(i)
					@time @save zipPath ρ
				end

				println("act zip ", i, "...")
				flush(stdout)
				@time for s = 1 : length(e)
					println(s)
					flush(stdout)
					oldPath = dataPathL * "rhov/rhov_" * string(i-1) * "_" * string(s) * ".jld2"
					path = dataPathL * "rhov/rhov_" * string(i) * "_" * string(s) * ".jld2"
					if !ispath(path)
						@load oldPath rhov
						@time rhov = ρ * rhov
						@save path rhov
						flush(stdout)
					end
				end
			end
			@save donePath donePath
			println()
			flush(stdout)
		end

		@time prepare!(L+1)

		donePath = dataPathL * "done/done_" * string(nev) * "_" * string(L+1) * ".jld2"
		if !ispath(donePath)
			detachPath = dataPathL * "detach.jld2"
			if ispath(detachPath)
				println("load detach...")
				flush(stdout)
				@time @load detachPath ρ
			else
				println("build detach...")
				flush(stdout)
				@time ρ = buildDetach()
				@time @save detachPath ρ
			end

			println("act detach...")
			flush(stdout)
			@time for s = 1 : length(e)
				println(s)
				flush(stdout)
				oldPath = dataPathL * "rhov/rhov_" * string(L) * "_" * string(s) * ".jld2"
				path = dataPathL * "rhov/rhov_" * string(L+1) * "_" * string(s) * ".jld2"
				if !ispath(path)
					@load oldPath rhov
					@time rhov = ρ * rhov
					@save path rhov
					flush(stdout)
				end
			end
		end
		@save donePath donePath

		u = Matrix{Float16}(undef, len, length(e))
		for s = 1 : length(e)
			path = dataPathL * "rhov/rhov_" * string(L+1) * "_" * string(s) * ".jld2"
			@time @load path rhov
			u[:,s] = rhov
		end
		println()
		flush(stdout)
	else
		println("prepare attach...")
		@time prepare!(0)
		ρ = LinearMap((C,B)->attach!(C,B),ziplen,len,ismutating=true,issymmetric=false,isposdef=false)
		println("act attach...")
		@time u = Matrix(ρ * u)
		println()
		flush(stdout)
		for i = 1 : L
			println("prepare zip ", i, "...")
			@time prepare!(i)
			ρ = LinearMap((C,B)->zip!(C,B,i),ziplen,ziplen,ismutating=true,issymmetric=false,isposdef=false)
			println("act zip ", i, "...")
			@time u = Matrix(ρ * u)
			println()
			flush(stdout)
		end
		println("prepare detach...")
		@time prepare!(L+1)
		ρ = LinearMap((C,B)->detach!(C,B),len,ziplen,ismutating=true,issymmetric=false,isposdef=false)
		println("act detach...")
		@time u = Matrix(ρ * u)
		println()
		flush(stdout)
	end

	return adjoint(v) * u
end


#=
Translation (lattice shift).
=#

function Tind(ind::Int64,L::Int64,right::Bool)
	if ind==2 && iseven(L)
		return 1
	end
	if ind==1 && iseven(L)
		return 2
	end
	state = (ind-1) & (4^L-1)
	if right
		return 1+(state>>2)+((state&3)<<(2*(L-1)))
	else
		return 1+(state<<2)&(4^L-1)+(state>>(2*(L-1)))
	end
end

function Tfunc!(C,B,L::Int64=L,right::Bool=true)
	Threads.@threads for preind = 1 : len
		ind = basis[preind]
		C[preind] = B[fromInd[Tind(ind,L,right)]]
	end
end

T=LinearMap((C,B)->Tfunc!(C,B),len,ismutating=true,issymmetric=false,isposdef=false)


#=
Simultaneous diagonalization and output.
=#

# Print as mathematica array to reuse Mathematica code for making plots.
function mathematicaVector(V)
	s="{"
	for i = 1:(size(V,1)-1)
		s*=string(V[i])
		s*=", "
	end
	s*=string(V[size(V,1)])
	s*="}"
	return s
end

# Print as mathematica matrix to reuse Mathematica code for making plots.
function mathematicaMatrix(M)
	s="{\n"
	for i = 1:(size(M,1)-1)
		s*=mathematicaVector(M[i])
		s*=",\n"
	end
	s*=mathematicaVector(M[size(M,1)])
	s*="\n}\n"
	return s
end

function diagonalizeHT(e,v,T)
	println("diagonalizing H,T...")
	println()
	flush(stdout)

	smallH = Matrix(diagm(e))
	smallT = Matrix(adjoint(v)*T*v)
	smalle,smallv = eigen(smallH+smallT)

	Hs = real(diag(adjoint(smallv)*smallH*smallv))
	Ts = complex(diag(adjoint(smallv)*smallT*smallv))
	Ps = real(map(log,Ts)/(2*π*im))*L
	HPs = hcat(Hs,Ps)
	HPs = sort([HPs[i,:] for i in 1:size(HPs, 1)])
	s=""
	s*=string(HPs[1][1])
	println(mathematicaMatrix(HPs))
end

function diagonalizeHTρ(e,v,T)
	println("diagonalizing H,T,ρ...")
	println()
	flush(stdout)

	smallH = Matrix(diagm(e))
	smallρ = ρMatrix(v)
	smallT = Matrix(adjoint(v)*T*v)
	smalle,smallv = eigen(smallH+smallT+smallρ)

	Hs = real(diag(adjoint(smallv)*smallH*smallv))
	Ts = complex(diag(adjoint(smallv)*smallT*smallv))
	Ps = real(map(log,Ts)/(2*π*im))*L
	ρs = real(diag(adjoint(smallv)*smallρ*smallv))
	HPρs = hcat(Hs,Ps,ρs)
	HPρs = sort([HPρs[i,:] for i in 1:size(HPρs, 1)])
	s=""
	s*=string(HPρs[1][1])
	println(mathematicaMatrix(HPρs))
end

if onlyT
	@time diagonalizeHT(e,v,T)
else
	@time diagonalizeHTρ(e,v,T)
end
