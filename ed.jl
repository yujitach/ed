using LinearAlgebra,LinearMaps
using Arpack

const L=12

#=

As Julia's array is 1-based, I use 

	"ind" = 1 ... 4^L

as the array index.

"state" is what you obtain by encoding edge labels using the following mapping,
and takes the values 0 ...  4^L-1.

=#

const mapping=Dict('x'=>0, '0'=>1, '+'=>2,'-'=>3)
const revmapping=Dict(0=>'x', 1=>'0', 2=>'+',3=>'-')

#=

The conversion between "ind" and "state" is basically done by considering modulo 4^L.

When L is even, however, some additional care is needed, 
since "xxx...x" can come with two "start labels".
My convention is to use
	ind == 4^L  for "x....x" with ρ as the start label
	ind == 1  for "x....x" with 1 as the start label

=#

function indexFromString(s::String)
	if length(s)!=L
		error("length doesn't match")
	end
	a=0
	for i in L : -1 : 1
		a<<=2
		a+=mapping[s[i]]
	end
	if a==0
		a=4^L
	end
	return a
end

function stringFromIndex(ind)
	if ind==4^L
		ind=0
	end
	s=""
	for i in 1 : L
		s*=revmapping[(ind&3)]
		ind>>=2
	end
	return s
end


function trailingXs(ind)
	i=0
	while ind!=0
		ind>>=2
		i+=1
	end
	L-i
end

const flagshift = L+2


function bitdump(i)
	s=""
	for pos = 0 : L-1
		if (i & (1<<pos))!=0
			s*="1"
		else
			s*="0"
		end
	end
	println(s)
	a = (i >> flagshift) & 3
	if(a==3)
		println("not allowed")
	else
		println("Z3 charge: $a")
	end
end


#=

flag[ind] is a bitmask; from the 0th bit to (L-1)-th bit , 1 indicates that
the edge labels (i+1) (i+2) are x,x and corresponds to ρ,1,ρ
the (L+2)th and (L+3)th bit combine to form 0,1,2,3 , 
where 0,1,2 are twisted Z3 charge and 3 means it is a forbidden state

=#

flag_ = zeros(Int32,4^L)

function setFlag!(flag,ind)
	state=ind
	if(ind==4^L)
		state=0
		if(isodd(L))
			# not allowed
			flag[ind] |= 3 << flagshift
			return
		end
	end
	evenxs=iseven(trailingXs(state))
	tot=0
	if(ind==1 && iseven(L))
		state=0
		evenxs=false
	end
	for pos = 0 : L-1
		a=(state >> (2*pos)) & 3
		if a==0 
			if(evenxs)
				flag[ind] |= 1<<pos
			end
			evenxs = ! evenxs
		else
			if(!evenxs)
				# not allowed
				flag[ind] |= 3 << flagshift
				return
			end
			if a==2
				tot+=1
			elseif a==3
				tot+=2
			end
		end
	end
	tot%=3
	if ind==0 && isodd(L)
		flag[ind] |= 3<<flagshift
		return
	end
	flag[ind] |= tot << flagshift
end

diag_ = zeros(Float64,4^L)

const U = 10	# suppression factor for forbidden state
const Z = 10	# suppression factor for states charged under twisted Z3
const ζ = (√(13)+3)/2
const ξ = 1/√ζ
const x = (2-√(13))/2
const z = (1+√(13))/2
const y1 = (5-√(13) - √(6+√(13)))/12
const y2 = (5-√(13) + √(6+√(13)))/12


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

function stateFromInd(ind)
	state=ind
	if ind==4^L
		state=0
	end
	if ind==1 && iseven(L)
		state=0
	end
	return state
end

function mainFlag(flag,ind)::Int32
	return (flag[ind] >>flagshift) & 3
end

function localStatePair(state,i)
	j=i+1
	if j==L
		j=1
	end
	a = (state >> (2*(i-1))) & 3
	b = (state >> (2*(j-1))) & 3
	return a,b
end

function isρ1ρ(flag,ind,i)
	return ((flag[ind] >> (i-1)) & 1) == 1
end

function computeDiag!(diag,flag,ind)
	state=stateFromInd(ind)
	fl = mainFlag(flag,ind)
	if fl==3
		diag[ind]=U
		return
	elseif fl==1 || fl==2
		diag[ind]=Z
		return
	end
	diag[ind]=0
	for i = 1 : L
		sp=localStatePair(state,i)
		if sp==sX0
			diag[ind] -= 1
		elseif sp==s0X
			diag[ind] -= 1
		elseif sp==sXX && isρ1ρ(flag,ind,i)
			diag[ind] -= 1/ζ
		elseif sp==sPM
			diag[ind] -= y1 * y1
		elseif sp==sMP
			diag[ind] -= y2 * y2
		elseif sp==s00
			diag[ind] -= x * x
		elseif sp==s0P
			diag[ind] -= y1 * y1
		elseif sp==sP0
			diag[ind] -= y2 * y2
		elseif sp==sMM
			diag[ind] -= z * z
		elseif sp==s0M
			diag[ind] -= y2 * y2
		elseif sp==sM0
			diag[ind] -= y1 * y1
		elseif sp==sPP
			diag[ind] -= z * z
		end
	end
end



function newInd(state,i,sp)
	(a,b)=sp
	
	state &= ~(3<<(2*(i-1)))
	state |= (a<<(2*(i-1)))
	
	j=i+1
	if j==L+1
		j=1
	end
	
	state &= ~(3<<(2*(j-1)))
	state |= (b<<(2*(j-1)))

	if(state!=0)
		return state
	end
	if(isodd(i))
		return 4^L
	else
		return 1
	end
end

function pettyPrint(v)
	for x in v
		if(abs(x)<.0001)
			x=0
		end
		print("$x,")
	end
	println("")
end

function Hfunc(B,diag,flag)
	C = diag .* B
	for ind = 1 : 4^L
		if  mainFlag(flag,ind) !=0
			break
		end
		state=stateFromInd(ind)
		for i = 1 : L
			sp=localStatePair(state,i)
			if sp==sXX  && isρ1ρ(flag,ind,i)
				C[newInd(state,i,sPM)] -= ξ * y1 * B[ind]
				C[newInd(state,i,sMP)] -= ξ * y2 * B[ind]
				C[newInd(state,i,s00)] -= ξ * x * B[ind]
			elseif sp==sPM
				C[newInd(state,i,sXX)] -= y1 * ξ * B[ind]
				C[newInd(state,i,sMP)] -= y1 * y2 * B[ind]
				C[newInd(state,i,s00)] -= y1 * x * B[ind]
			elseif sp==sMP
				C[newInd(state,i,sXX)] -= y2 * ξ * B[ind]
				C[newInd(state,i,sPM)] -= y2 * y1 * B[ind]
				C[newInd(state,i,s00)] -= y2 * x * B[ind]
			elseif sp==s00
				C[newInd(state,i,sXX)] -= x * ξ * B[ind]
				C[newInd(state,i,sPM)] -= x * y1 * B[ind]
				C[newInd(state,i,sMP)] -= x * y2 * B[ind]
			elseif sp==s0P
				C[newInd(state,i,sP0)] -= y1 * y2 * B[ind]
				C[newInd(state,i,sMM)] -= y1 * z * B[ind]
			elseif sp==sP0
				C[newInd(state,i,s0P)] -= y2 * y1 * B[ind]
				C[newInd(state,i,sMM)] -= y2 * z * B[ind]
			elseif sp==sMM
				C[newInd(state,i,s0P)] -= z * y1 * B[ind]
				C[newInd(state,i,sP0)] -= z * y2 * B[ind]
			elseif sp==s0M
				C[newInd(state,i,sM0)] -= y2 * y1 * B[ind]
				C[newInd(state,i,sPP)] -= y2 * z * B[ind]
			elseif sp==sM0
				C[newInd(state,i,s0M)] -= y1 * y2 * B[ind]
				C[newInd(state,i,sPP)] -= y1 * z * B[ind]
			elseif sp==sPP
				C[newInd(state,i,s0M)] -= z * y2 * B[ind]
				C[newInd(state,i,sM0)] -= z * y1 * B[ind]				
			end
		end
	end
	return C
end

println("preparing...")
for i = 1 : 4^L
	setFlag!(flag_,i)
	computeDiag!(diag_,flag_,i)
end

println("computing eigenvalues...")
H=LinearMap(B->Hfunc(B,diag_,flag_),4^L,ismutating=false,issymmetric=true,isposdef=false)
@time e,v = eigs(H,nev=8,which=:SR)
println(sort(e))