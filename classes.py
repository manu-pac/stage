## Code taken from TD

# Checks that `o` is an instance of `t` (ex: integer, list). Produces a clear error message otherwise.
# This function is not essential but can help a lot for debugging.
def check_type(o, t, name=None):
	if(name is None): name = "[no name]"
	assert isinstance(o, t), (f"Type problem: variable {name} (type: {type(o)}; value: {o}) is not an instance of {t}")

# For interpretation functions.
class InterpretationFunc:
    # true_ps: set of strings
    def __init__(self, true_ps):
        check_type(true_ps, set, "true_ps")

        self._true_ps = true_ps

    # Remark: __call__ can be called using the ()-notation: "i(p)" is translated as "i.__call__(p)". Use the ()-notation instead of calling __call__ explicitly.
    # Returns the interpretation of `p`.
    # p: string
    def __call__(self, p):
        check_type(p, str, "p")

        return (p in self._true_ps)

    # Returns a string representation of the object. Used to print the object in a readable way.
    def __str__(self):
        return str(self._true_ps)

    # Returns a string representation of the object. Also used to print the object in a readable way.
    def __repr__(self):
        return str(self)

# The general class for logical formulas.
# This class is sub-classed below.
class Formula:
    pass;

# For atomic formulas (i.e. that are composed of a single propositional letter only).
class PLetter(Formula): # This means that PLetter is a subclass of Formula (i.e. that any instance of PLetter should be considered an instance of Formula).
    # p: string
    def __init__(self, p):
        check_type(p, str, "p")

        self._p = p

    # Checks whether the formula is true according to the interpretation function `i_func`. This should mirror the definition of the valuation of PL.
    # i_func: InterpretationFunc
    def check(self, i_func):
        check_type(i_func, InterpretationFunc, "i_func")

        return i_func(self._p)

    # Returns the list of all (minimal) partial interpretation functions for which the valuation of the formula is the boolean value `value`.
    # If `value` is not specified, the default value True is used.
    def build(self, value=True):
        check_type(value, bool, "value")

        return [PartialInterpretationFunc({self._p:value})]

    # Returns a string representation of the object. Used to print the object in a readable way.
    def __str__(self):
        return self._p

    # Returns a string representation of the object. Also used to print the object in a readable way.
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # Two PLetter objects are equal if they are both PLetter instances
        # and their internal propositional letter string (_p) is the same.
        return isinstance(other, PLetter) and self._p == other._p

    def __hash__(self):
        # The hash of a PLetter should be based on its propositional letter string.
        return hash(self._p)

# Negation
class Neg(Formula):
	# phi: Formula
	def __init__(self, phi):
		check_type(phi, Formula, "phi")

		self._phi = phi;

	# Checks whether the formula is true according to the interpretation function `i_func`. This should mirror the definition of the valuation of PL.
	# i_func: InterpretationFunc
	def check(self, i_func):
		check_type(i_func, InterpretationFunc, "i_func")

		return not self._phi.check(i_func)

	# Returns the list of all (minimal) partial interpretation functions for which the valuation of the formula is the boolean value `value`.
	# If `value` is not specified, the default value True is used.
	def build(self, value=True):
		check_type(value, bool, "value")

		return self._phi.build(not value)

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return f'(¬{self._phi})'

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)


	def __eq__(self, other):
			# Two Neg objects are equal if they are both Neg instances
			# and their negated formula (_phi) is equal.
			return isinstance(other, Neg) and self._phi == other._phi

	def __hash__(self):
			# The hash of a Neg formula should be based on the hash of its subformula (_phi).
			return hash(self._phi)

# Conjunction
class Conj(Formula):
	# phi: Formula
	# psi: Formula
	def __init__(self, phi, psi):
		check_type(phi, Formula, "phi")
		check_type(psi, Formula, "psi")

		self._phi = phi;
		self._psi = psi;

	# Checks whether the formula is true according to the interpretation function `i_func`. This should mirror the definition of the valuation of PL.
	# i_func: InterpretationFunc
	def check(self, i_func):
		check_type(i_func, InterpretationFunc, "i_func")

		return self._phi.check(i_func) and self._psi.check(i_func)

	# Returns the list of all (minimal) partial interpretation functions for which the valuation of the formula is the boolean value `value`.
	# If `value` is not specified, the default value True is used.
	def build(self, value=True):
		check_type(value, bool, "value")
		if value:
			result = []
			for f1 in self._phi.build(True):
				for f2 in self._psi.build(True):
					result.append(f1.merge(f2))
			return result
		else:
			return self._phi.build(False) + self._psi.build(False)

	# Returns a string representation of the object. Used to print the object in a readable way.
	def __str__(self):
		return f"({self._phi} ∧ {self._psi})"

	# Returns a string representation of the object. Also used to print the object in a readable way.
	def __repr__(self):
		return str(self)

	def __eq__(self, other):
			# Two Conj objects are equal if they are both Conj instances
			# and both their left (_phi) and right (_psi) conjuncts are equal.
			return isinstance(other, Conj) and self._phi == other._phi and self._psi == other._psi

	def __hash__(self):
			# The hash of a Conj formula should be based on the hashes of both its subformulas.
			# Hashing a tuple of hashes is a common way to combine them.
			return hash((self._phi, self._psi))