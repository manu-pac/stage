# Checks that `o` is an instance of `t` (ex: integer, list).
# Produces a clear error message otherwise.
# This function is not essential but can help a lot for debugging.
def check_type(o, t, name=None):
	if(name is None): name = "[no name]"
	assert isinstance(o, t), (f"Type problem: variable {name} (type: {type(o)}; value: {o}) is not an instance of {t}")

# Constant
class C:
	# name: string
	def __init__(self, name):
		check_type(name, str, "name")

		self._name = name

	# Defines the behaviour of "==".
	# In this case: two C·s are considered equal if they have the same `_name`.
	def __eq__(self, other):
		return isinstance(other, C) and self._name == other._name

	# Required to be able to use the class in sets or dictionaries.
	def __hash__(self):
		return hash(self._name)

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return self._name

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)

# Variable
class V:
	# name: string
	def __init__(self, name):
		check_type(name, str, "name")

		self._name = name

	# Defines the behaviour of "==".
	# In this case: two V·s are considered equal if they have the same `_name`.
	def __eq__(self, other):
		return isinstance(other, V) and self._name == other._name

	# Required to be able to use the class in sets or dictionaries.
	def __hash__(self):
		return hash(self._name)

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return self._name

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)


# Predicate
class P:
	# name: string
	def __init__(self, name, arity):
		check_type(name, str, "name")
		check_type(arity, int, "arity")

		self._name = name
		self.arity = arity

	# Defines the behaviour of "==".
	# In this case: two P·s are considered equal if they have the same `_name` and the same `arity`.
	def __eq__(self, other):
		return isinstance(other, P) and (self._name == other._name) and (self.arity == other.arity)

	# Required to be able to use the class in sets or dictionaries.
	def __hash__(self):
		return hash((self._name, self.arity))

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return self._name

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)


class InterpretationFunc:
	# c_dic: dictionary; keys are C·s, values are integers
	# p_dic: dictionary; keys are P·s, values are sets of tuples of integers
	def __init__(self, c_dic, p_dic):
		self._c_dic = c_dic
		self._p_dic = p_dic

	# Returns a boolean that indicates whether this interpretation function is an interpretation function on a given domain.
	def compatible(self, domain):
		check_type(domain, set)

		elements = set()
		for item in self._c_dic:
			elements.add(self._c_dic[item])

		for item in self._p_dic:
			for subitem in self._p_dic[item]:
				for subsubitem in subitem:
					elements.add(subsubitem)

		return (elements - domain) == set()

	# Remark: __getitem__ can be called using the []-notation: "i[x]" is translated as "i.__getitem__(x)". Use the []-notation instead of calling __getitem__ explicitly.
	# Returns the interpretation of `x`.
	# x: either a C or a P
	def __getitem__(self, x):
		if(isinstance(x, C)): return self._c_dic[x] # Raises an exception if the constant has no entry in `_c_dic`.
		if(isinstance(x, P)): return self._p_dic.get(x, set()) # Returns an empty set if the predicate has no entry in `_p_dic`.
		raise TypeError

	# Returns the list obtained from `l` by replacing all constants by their interpretation (other elements should appear unaffected).
	# The original list `l` should not be affected.
	# (Be aware that this function returns a list and not a tuple. If you need a tuple, use the `tuple` function to convert the list into one.)
	# l: list of C·s and V·s
	def map(self, l):
		check_type(l, list, "l")
		mapped = []
		for item in l:
			if isinstance(item,C):
				mapped.append(self._c_dic[item])
			else:
				mapped.append(item)
		return mapped


	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		tmp = list(self._c_dic.items())
		tmp.extend(self._p_dic.items())
		s = ', '.join([f"{k}: {v}" for (k, v) in tmp])
		return f'{s}'

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)

class Model:
	# domain: set of integers
	# i_func: InterpretationFunc
	def __init__(self, domain, i_func):
		check_type(domain, set, "domain")
		check_type(i_func, InterpretationFunc, "i_func")
		assert i_func.compatible(domain)

		self.domain = domain
		self.i_func = i_func

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return f'{{D={self.domain}; I={self.i_func}}}'

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)

# For variable assignments.
class VarAssignment:
	# dic: dictionary; keys are Vs, values are integers
	# If `dic` is not specified, the empty dictionary ({}) is used.
	def __init__(self, dic={}):
		check_type(dic, dict, "dic")

		self._dic = dic

	# Returns the variable assignment that only differ from the present one (i.e. `self`) with "x := d".
	# The present assignment is not modified and a new assignment is instantiated.
	# x: V
	# d: integer
	def assign(self, x, d):
		check_type(x, V, "x")
		check_type(d, int, "d")

		newdic = dict(self._dic) # creates copy
		newdic[x] = d

		return VarAssignment(newdic)

	# Returns the list obtained from `l` by replacing all variables by their assignments (other elements should appear unaffected).
	# The original list `l` should not be affected.
	# (Be aware that this function returns a list and not a tuple. If you need a tuple, use the `tuple` function to convert the list into one.)
	# l: list
	def map(self, l):
		check_type(l, list, "l")

		newlist = []
		for item in l:
			if isinstance(item,V):
				newlist.append(self._dic[item])
			else:
				newlist.append(item)
		return newlist


	# Returns a string representation of the object. Used to print the object in a nice way.
	def __str__(self):
		return f'{self._dic}'

# The general class for logical formulas.
# This class is sub-classed below.
class Formula:
	# Checks whether the formula is true according to the model `m`.
	# The use of this method requires that the formula be closed.
	# This method does almost nothing by itself. All the work is done by the `check` method defined for each kind of formulas (sub-classes of `Formula`).
	# m: Model
	def check_closed(self, m):
		check_type(m, Model, "m")

		f = VarAssignment() # Empty partial variable assignment.
		return self.check(m, f)

# Predicate application
class PredApp(Formula):
	# pred: P
	# args: list of V·s and C·s
	def __init__(self, pred, args):
		check_type(pred, P, "pred")
		assert (pred.arity == len(args)), f"{pred.arity} argument·s expected but {len(args)} given."
		check_type(args, list, "args")

		self._pred = pred
		self._args = args

	# Checks whether the formula is true according to the model `m` and the variable assignment `f`.
	# m: Model
	# f: VarAssignment
	def check(self, m, f):
		check_type(m, Model, "m")
		check_type(f, VarAssignment, "f")

		ents = tuple(m.i_func.map(f.map(self._args))) # Returns list with all constants and variables substituted by their elments on the domain

		return ents in m.i_func[self._pred]

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return f"{self._pred}({','.join([str(x) for x in self._args])})"


# Negation application
class Neg(Formula):
  # Receives a formula as input
  def __init__(self,phi):
    check_type(phi,Formula,"phi")

    self._phi = phi

  def check(self, m, f):
    check_type(m,Model,"m")
    check_type(f,VarAssignment,"f")

    return not self._phi.check(m,f)

  # Returns a string representation of the object. Used to print the object in a readable way
  def __str__(self):
    return f'(¬{self._phi})'

class Ex(Formula):
  def __init__(self,v,phi):
    check_type(v,V,"v")
    check_type(phi,Formula,"phi")
    self._phi = phi
    self._v = v

  def check(self,m,f):
    check_type(m,Model,"m")
    check_type(f,VarAssignment,"f")

    for item in m.domain:
      if self._phi.check(m,f.assign(self._v,item)):
        return True
    return False

  def __str__(self):
      return f'(∃{self._v}{self._phi})'


class Conj(Formula):
  def __init__(self,phi,psi):
    check_type(phi,Formula,"phi")
    check_type(psi,Formula,"psi")

    self._phi=phi
    self._psi=psi

  def check(self,m,f):
    check_type(m,Model,"m")
    check_type(f,VarAssignment,"f")

    return self._phi.check(m,f) and self._psi.check(m,f)

  def __str__(self):
      return f'({self._phi}∧{self._psi})'