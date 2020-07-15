"""
This file contains samples of nonsense pseudocode listings which excercise many
(all?) of the syntax features and may serve as good integration tests for
utilities which process pseudocode.
"""


__all__ = [
    "MINIMAL",
    "ALL_SEMANTIC_FEATURES",
    "WHITESPACE_AND_COMMENTS",
]


MINIMAL = "f():f()"

ALL_SEMANTIC_FEATURES = """
        func_no_args():
            return 0

        func_with_arg(a):
            return a

        func_with_multiple_args(a, b, c):
            return a + b + c

        if_else_stmts(a, b, c, d):
            # Minimal
            if (0):
                return a

            # If-else
            if (0):
                return a
            else:
                return b

            # If-elseif
            if (0):
                return a
            else if (1):
                return b

            # If-multiple-elseif
            if (0):
                return a
            else if (1):
                return b
            else if (2):
                return c

            # If-multiple-elseif-else
            if (0):
                return a
            else if (1):
                return b
            else if (2):
                return c
            else:
                return d

            # Multiple lines in statement
            if (0):
                x = {}
                return x
            else if (1):
                y = {}
                return y
            else:
                z = {}
                return z

            # Names defined in 'if' body continue to be defined outside
            # (The following fails if any are considered labels)
            x[1] = 100
            y[2] = 200
            z[3] = 300

        for_each_stmts(a, b, c):
            # Single value
            for each w in a:
                return w

            # Multiple values
            for each x in a, b, c:
                return x

            # Values are expressions
            for each y in a + 1, b + 2, c + 3:
                return y

            # Multi-line body
            for each z in a, b, c:
                z[1] = 100
                d = {}

            # Ensure names defined by/in for-each remain defined (the following
            # would fail if not)
            w[0] = 0
            x[1] = 100
            y[2] = 200
            z[3] = 300
            d[4] = 400

        for_stmts(a, b):
            # Endpoints are constants
            for x = 1 to 3:
                return x

            # Endpoints are expressions
            for y = a + 1 to b + 2:
                return y

            # Multi-line body
            for z = 1 to 3:
                a += z
                c = {}

            # Ensure names defined by/in for remain defined (the following
            # would fail if not)
            x[1] = 100
            y[2] = 200
            z[3] = 300
            c[4] = 400

        while_stmts(a, b):
            # Constant expression
            while (False):
                return 0

            # Compound expression
            while (a + b > 0):
                return 1

            # Multi-line body
            while (True):
                a += b
                c = {}

            # Ensure names defined in while remain defined (the following would
            # fail if not)
            c[1] = 100

        assignment_stmts(a, b):
            # Simple assignment
            a = 100

            # Assignment of expression
            a = b + 1

            # Assignment to array/map
            a[0] = 100
            a[1][foo][b + 2][bar] = 200

            # Compound assignments
            a += 100
            a -= 200
            a *= 300
            a //= 400
            a **= 500
            a &= 600
            a ^= 700
            a |= 800
            a <<= 800
            a >>= 800

        label_discovery():
            # NB: Many of the label names in this function are used as variable
            # names in other functions, ensuring that we're starting a new
            # scope for each
            if (a): foo(b)
            else if (c): foo(d)
            else: foo(e)

            for each _ in f, g: foo(h)

            for _ = i to j: foo(k)

            while (l): foo(m)

            map = {}
            map[n] = o + map[q]

            return v

        return_stmts(a):
            # Simple value
            return a

            # Expression
            return a + 1

        function_call_stmts(a, b):
            foo()
            foo(a)
            foo(a,)
            foo(a, b)
            foo(a, b,)
            foo(a + 1, b + 2)

        unary_expressions(a):
            r = {}
            r[0] = -a
            r[1] = +a
            r[2] = ~a
            r[3] = not a

        binary_expressions(a, b):
            r = {}

            # Logical
            r[0] = a or b
            r[1] = a and b

            # Comparision
            r[2] = a == b
            r[3] = a != b
            r[4] = a <= b
            r[5] = a >= b
            r[6] = a < b
            r[7] = a > b

            # Bit-wise
            r[8] = a | b
            r[9] = a ^ b
            r[10] = a & b

            # Shift
            r[11] = a << b
            r[12] = a >> b

            # Arithmetic
            r[13] = a + b
            r[14] = a - b
            r[15] = a * b
            r[16] = a // b
            r[17] = a % b
            r[18] = a ** b

            # Parentheses
            r[19] = (a)
            r[20] = (a + b)
            r[20] = (a + b) + 3
            r[20] = ((((a)) + (b)))

        atom_expressions(a, b):
            # Empty map
            r = {}

            # Function call
            r[0] = foo()
            r[1] = foo(a)
            r[2] = foo(a,)
            r[3] = foo(a, b)
            r[4] = foo(a, b,)
            r[5] = foo(a + 1, b + 2)

            # Variable
            r[6] = a

            # Subscripted variable
            r[7] = r[100]
            r[9] = r[a_label]
            r[9] = r[1 + 2]
            r[10] = r[100][a_label][1 + 2]

            # Label
            r[11] = i_am_a_label

            # Boolean
            r[12] = True
            r[13] = False

            # Zeros
            r[14] = 0
            r[15] = 0b0
            r[16] = 0x0
            # Numbers with all digits
            r[17] = 1234567890
            r[18] = 0b1010
            r[19] = 0x1234567890abcdefABCDEF
            # Leading zeros
            r[20] = 00000000000000001234567890
            r[21] = 0b000000000000000000001010
            r[22] = 0x001234567890abcdefABCDEF
    """


WHITESPACE_AND_COMMENTS = """
        # Leading comment at start
        # With adjacent second line

        # And non-adjacent third line


        # And very non-adjacent fourth line and then a space...

        foo():
            return 0

        # And spaced between functions

        bar():
            return 0

        # And adjacent to a function
        baz():
            return 0

        multi_line_stmts():  # And on a function definition
            # Before an if
            if (True):  # And on an if
                # And inside an if
                foo()
            # And before an else if
            else if (True):  # And on an else if
                # And inside an else if
                foo()
            # And before an else
            else:  # And on an else
                # And inside an else
                foo()

            # Before a for each
            for each x in 1, 2, 3:  # On a for each
                # Inside a for each
                foo()

            # Before a for
            for x = 1 to 3:  # On a for
                # Inside a for
                foo()

            # Before a while
            while (False):  # On a while
                # Inside a while
                foo()

            # Before an assignment
            x = 100  # On an assignment

            # Before a call
            foo()  # On a call

            # Before a return
            return 0  # On a return

        spaced_before_multi_line_stmts():
            # Spaced before an if

            if (True):
                foo()
            # And spaced before an else if

            else if (True):
                foo()
            # And spaced before an else

            else:
                foo()

            # Spaced before a for each

            for each x in 1, 2, 3:
                foo()

            # Spaced before a for
            for x = 1 to 3:
                foo()

            # Spaced before a while
            while (False):
                foo()

            # Spaced before an assignment

            x = 100

            # Spaced before a call

            foo()

            # Spaced before a return

            return 0

        one_liner_function(): return 0  # Comment on one-liner function

        compact_expressions():
            if (True): return 0  # Comment on one-liner if
            else if (True): return 0  # Comment on one-liner else if
            else: return 0  # Comment on one-liner else

            if (True): return 0  # Comment on one-liner if

            else if (True): return 0  # Comment on one-liner else if

            else: return 0  # Comment on one-liner else

            for each x in 1, 2, 3: foo()  # Comment on one-liner for each

            for x = 1 to 3: foo()  # Comment on one-liner for

            while (False): foo()  # Comment on one-liner while

        unusual_spacing  ( a  ,b ) :

            if(  True ) :

              foo()

            else     if(  False ) :

             baz()

            else  :

                    bar()


            for   each  x   in  3  ,  4,5  :

                foo()

            for x=3   to  4  :

               foo()


            while(  True ) :

                 foo()


            if(  True ) :foo()

            else     if(  False ) :baz()


            else  :bar()


            for   each  x   in  3  ,  4,5  :foo()


            for x=3   to  4  :foo()


            while(  True ) :foo()


            a=-b
            a[b]=c+d
            a    =    -   b
            a  [  b  ]  =  c  +  c

            foo  ( )
            foo  ( 1 )
            foo  ( 1,3 )

            m = {  }

            return     1

            no_space()#before comment

          # Comment indentation
        is_irrelevant():   # as
              # this
            simple()  # demo
             # Shows
          # There we go

        # And at the end of a file
        # Adjacent

        # And non-adjacent


        # And very non-adjacent
    """
