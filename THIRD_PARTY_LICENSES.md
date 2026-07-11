# Third-party licenses

This repository vendors or depends on the following third-party components. The
`Hunyuan3D-Swift` source code itself is MIT-licensed (see `LICENSE`).

### xatlas
- Location: `Sources/CXatlas/` (`xatlas.cpp`, `xatlas.h`) with a thin C shim.
- Upstream: <https://github.com/jpcy/xatlas>
- License: **MIT**

```
Copyright (c) 2018 Jonathan Young

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the above copyright notice and this permission notice being
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

xatlas additionally incorporates work under permissive licenses (e.g. portions
derived from `thekla_atlas`, MIT). See the header of `Sources/CXatlas/xatlas.cpp`
for the full upstream notices.

## Swift package dependencies

### mlx-swift
- Upstream: <https://github.com/ml-explore/mlx-swift> (pinned to 0.31.4)
- License: **MIT** (Copyright © 2023–2024 Apple Inc.)
- Transitively pulls `swift-numerics` (Apache-2.0 with the Runtime Library
  Exception).
