import sys, re

# Patch TritonTemplate.__init__ assertion
p1 = '/root/miniconda3/lib/python3.12/site-packages/torch/_inductor/select_algorithm.py'
with open(p1, 'r') as f:
    c = f.read()

# Fix 1: TritonTemplate duplicate name assertion
c = c.replace(
    '        assert name not in self.all_templates, "duplicate template name"\n        TritonTemplate.all_templates[name] = self\n',
    '        if name not in self.all_templates:\n            TritonTemplate.all_templates[name] = self\n'
)

# Fix 2: ExternKernelChoice duplicate assertion (line ~3097)
c = c.replace(
    'assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"\n    register_extern_kernel(name)',
    '# patched\n    register_extern_kernel(name)'
)

with open(p1, 'w') as f:
    f.write(c)
print('patched select_algorithm.py')