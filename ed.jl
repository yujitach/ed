const mapping=Dict('x'=>0, '0'=>1, '+'=>2,'-'=>3)
const revmapping=Dict(0=>'x', 1=>'0', 2=>'+',3=>'-')
const L=6



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

#=

When L is even, 
the non-allowed state ind=1 ( 0x^(L-1) ) is reused as x^L with start label 1, while
the state 4^+ (x^L) is x^L with start label ρ.

When L is odd, no special treatment is done.

flag[ind] is a bitmask; from the 0th bit to (L-1)-th bit , 1 indicates that
the edge labels (i+1) (i+2) are x,x and corresponds to ρ,1,ρ
the (L+2)th and (L+3)th bit combine to form 0,1,2,3 , 
where 0,1,2 are twisted Z3 charge and 3 means it is a forbidden state

=#

function bitdump(ind)
	i=flag[ind]
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

flag = zeros(Int16,4^L)

function setFlag(ind)
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

diag = zeros(Float64,4^L)

const U = 10	# suppression factor for forbidden state
const Z = 10	# suppression factor for states charged under twisted Z3
const ζ = (√(13)+3)/2
const ξ = 1/√ζ
const x = (2-√(13))/2
const z = (2-√(13))/2
const y1 = (5-√(13) - √(6+√(13)))/12
const y2 = (5-√(13) + √(6+√(13)))/12

function computeDiag(ind)
	state=ind
	if ind==4^L
		state=0
	end
	if ind==1 && iseven(L)
		state=0
	end
	fl = (flag[ind] >>flagshift) & 3
	if fl==3
		diag[ind]=U
		return
	else fl==1 || fl==2
		diag[ind]=Z
		return
	end
	for i = 1 : L-1
		j = i+1
		if j==L
			j=1
		end
		a = (state >> 2*(i-1)) & 3
		b = (state >> 2*(j-1)) & 3
		if a==0 && b==1
			diag[ind] += 1
		elseif a==1 && b==0
			diag[ind] += 1
		elseif a==0 && b==0 && ((flag[ind] >> (i-1)) & 1 == 1)
			diag[ind] += 1/ζ
		elseif a==2 && b==3
			diag[ind] += y1 * y1
		elseif a==3 && b==2
			diag[ind] += y2 * y2
		elseif a==1 && b==1
			diag[ind] += x * x
		elseif a==1 && b==2
			diag[ind] += y1 * y1
		elseif a==2 && b==1
			diag[ind] += y2 * y2
		elseif a==3 && b==3
			diag[ind] += z * z
		elseif a==1 && b==3
			diag[ind] += y2 * y2
		elseif a==3 && b==1
			diag[ind] += y1 * y1
		elseif a==2 && b==2
			diag[ind] += z * z
		end
	end
end



for i = 1 : 4^L
	setFlag(i)
	computeDiag(i)
end

function replace(ind,i,a,b)
	ind &= ~(3<<(2*(i-1)))
	ind |= (a<<(2*(i-1)))
	i+=1
	if(i>L)
		i=1
	end
	ind &= ~(3<<(2*(i-1)))
	ind |= (b<<(2*(i-1)))
	return ind
end
