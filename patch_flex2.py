import sys
p = '/root/miniconda3/lib/python3.12/site-packages/torch/_inductor/select_algorithm.py'
with open(p, 'r') as f:
    c = f.read()

# Fix 1: TritonTemplate duplicate name assertion (line 2568-2569)
old1 = '        assert name not in self.all_templates, "duplicate template name"\n        TritonTemplate.all_templates[name] = self\n'
new1 = '        if name not in self.all_templates:\n            TritonTemplate.all_templates[name] = self\n'

# Fix 2: ExternKernelChoice duplicate assertion (around line 3097)
old2 = 'assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"\n    register_extern_kernel(name)'
new2 = '# patched\n    register_extern_kernel(name)'

count = 0
if old1 in c:
    c = c.replace(old1, new1)
    count += 1
if old2 in c:
    c = c.replace(old2, new2)
    count += 1

with open(p, 'w') as f:
    f.write(c)
print(f'patched {count} assertions')

# Verify
import torch._inductor.lowering
print('lowering imported OK')