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
	return a
end

function stringFromIndex(ind)
	s=""
	for i in 1 : L
		s*=revmapping[(ind&3)]
		ind>>=2
	end
	return s
end

function twistedZ3charge(ind)
	tot=0
	for i in 1 : L
		c=ind&3
		if(c==2)
			tot+=1
		elseif(c==3)
			tot+=2
		end
		ind>>=2
	end
	tot%3
end

function trailingXs(ind)
	i=0
	while ind!=0
		ind>>=2
		i+=1
	end
	L-i
end

function allowed(ind)
	if(iseven(ind))
		if(isodd(trailingXs(ind)+(trailing_zeros(ind)>>1) ))
			return false
		end
	end
	ind=ind>>trailing_zeros(ind)
	while ind!=0
		if(iseven(ind))
			if(isodd(trailing_zeros(ind)>>1))
				return false
			end
			ind >>= trailing_zeros(ind)
		else
			ind >>= 2
		end
	end
	return true
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

println(stringFromIndex(indexFromString("xx0+-x")))
println(trailingXs(indexFromString("xx0+-x")))
println(trailingXs(indexFromString("xxx+xx")))
println(trailingXs(indexFromString("xxxx-x")))
println(allowed(indexFromString("x++-+x")))
println(stringFromIndex(replace(indexFromString("x++-+x"),6,1,3)))
