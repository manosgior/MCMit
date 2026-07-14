import pyparsing as pp

# define grammar
vartok = pp.Word(pp.alphanums + '_')
simple_expr = pp.Forward()
simple_expr <<= vartok('lhs') + pp.Opt(pp.one_of('+ -')('op') + pp.Group(simple_expr)('rhs'))

# variable with slicing
slicetok = pp.Word(pp.alphanums + '_')('var') + \
        pp.Group(pp.Opt(pp.Literal('[') + simple_expr + pp.Literal(']')))('slice_exp')
# expr with slicing
full_expr = pp.Forward()
full_expr <<= pp.Group(slicetok)('lhs') + pp.Opt(pp.one_of('+ -')('op') + pp.Group(full_expr)('rhs'))

def parse_genblock(body: dict):
    genvar = body['var']
    genrange = [int(i) for i in body['range'].split(',')]
    expr = body['expr'].split('<=')
    lhs = expr[0].replace('.', '_').strip()
    rhs = expr[1].replace('.', '_').strip()
 
    return genvar, genrange, lhs, rhs

def append_to_operands(expr: str, mod_str: str):
    """
    Appends `mod_str` to every operand in expr. For example,
    if the expression is `a + b` and mod_str = '.q', the
    expression `a.q + b.q` is returned.
    """
    parsed_expr = full_expr.parse_string(expr, parse_all=True)

    def _append(parsed_expr):
        lhs = parsed_to_str(parsed_expr['lhs']) + mod_str
        if 'rhs' in parsed_expr:
            return lhs + ' ' + parsed_expr['op'] + ' ' + _append(parsed_expr['rhs'])
        else:
            return lhs

    return _append(parsed_expr)

def parsed_to_str(parsed_expr: pp.ParseResults):

    def _to_str(expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, list):
            return ''.join([_to_str(elem) for elem in expr])
    return _to_str(parsed_expr.as_list())

def eval_expr(expr: str | pp.ParseResults, genvar: str, genval: int) -> int:

    def _eval_expr(expr: dict):
        value = int(expr['lhs'].replace(genvar, str(genval)))
        if 'op' in expr.keys():
            if expr['op'] == '+':
                return value + _eval_expr(expr['rhs'])
            elif expr['op'] == '-':
                return value - _eval_expr(expr['rhs'])
            else:
                raise Exception(f'unsupported op: {expr["op"]}')
        else:
            return value
    
    if isinstance(expr, str):
        expr = simple_expr.parse_string(expr, parse_all=True)
    return _eval_expr(expr.as_dict())

def eval_slices(expr: str, genvar: str, genval: int):
    """
    Evaluate expressions inside slices according to the provided
    `genvar` and `genval`. For example, for the expression:
        `drive[i] + dc[i+1]`, `genvar = i`, `genval = 2`,
    the expression:
        `drive[2] + dc[3]`
    is returned
    """

    def _eval_expr(expr: pp.ParseResults):
        lhs = expr['lhs']['var']
        if len(expr['lhs']['slice_exp']) > 0:
            lhs += f"[{eval_expr(expr['lhs']['slice_exp'], genvar, genval)}]"

        if 'op' in expr.keys():
            return lhs + expr['op'] + _eval_expr(expr['rhs'])

        else:
            return lhs

    return _eval_expr(full_expr.parse_string(expr, parse_all=True))

