### 주 대비 (B=512, V1_SCALE_CP − control, 15 models × 2 t × 100 replicates/arch, 동시 CI .05/32 분위)

| arch | control | primary | Δ̂ | CI 하한 | CI 상한 | 하한>0 | 상한≤0 |
|---|---|---|---:|---:|---:|---|---|
| resnet18 | RATIO | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | MASS_CP | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | HOEFFDING_H | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | DIRECT_BETTING_H | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | SCALE_H | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | CP_IUT_H | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | COMPOSED_CP_BETTING_H | 1 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| resnet18 | GEOMETRY_DIRECT_CP | 0 | +0.00000 | +0.00000 | +0.00000 | N | Y |
| convnext_tiny | RATIO | 1 | +0.00104 | +0.00014 | +0.00255 | Y | N |
| convnext_tiny | MASS_CP | 1 | +0.00104 | +0.00014 | +0.00255 | Y | N |
| convnext_tiny | HOEFFDING_H | 1 | +0.00104 | +0.00014 | +0.00255 | Y | N |
| convnext_tiny | DIRECT_BETTING_H | 1 | -0.00035 | -0.00103 | +0.00018 | N | N |
| convnext_tiny | SCALE_H | 1 | +0.00013 | +0.00000 | +0.00045 | N | N |
| convnext_tiny | CP_IUT_H | 1 | +0.00073 | +0.00014 | +0.00155 | Y | N |
| convnext_tiny | COMPOSED_CP_BETTING_H | 1 | +0.00032 | +0.00000 | +0.00079 | N | N |
| convnext_tiny | GEOMETRY_DIRECT_CP | 0 | -0.00048 | -0.00112 | -0.00002 | N | Y |

### V1(PACS, 잠금값)과 S1 v2(DomainNet-6) 서술적 비교 — 검정 아님

| arch | control | V1 Δ̂ [CI] | S1 v2 Δ̂ [CI] |
|---|---|---|---|
| resnet18 | RATIO | +0.00876 [+0.00649, +0.01118] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | MASS_CP | +0.00876 [+0.00649, +0.01118] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | HOEFFDING_H | +0.00876 [+0.00649, +0.01118] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | DIRECT_BETTING_H | +0.00034 [-0.00065, +0.00131] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | SCALE_H | +0.00139 [+0.00071, +0.00219] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | CP_IUT_H | +0.00547 [+0.00390, +0.00716] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | COMPOSED_CP_BETTING_H | +0.00295 [+0.00195, +0.00404] | +0.00000 [+0.00000, +0.00000] |
| resnet18 | GEOMETRY_DIRECT_CP | -0.00105 [-0.00180, -0.00046] | +0.00000 [+0.00000, +0.00000] |
| convnext_tiny | RATIO | +0.06301 [+0.05568, +0.07033] | +0.00104 [+0.00014, +0.00255] |
| convnext_tiny | MASS_CP | +0.07800 [+0.06900, +0.08668] | +0.00104 [+0.00014, +0.00255] |
| convnext_tiny | HOEFFDING_H | +0.09084 [+0.08139, +0.10050] | +0.00104 [+0.00014, +0.00255] |
| convnext_tiny | DIRECT_BETTING_H | +0.02348 [+0.01935, +0.02785] | -0.00035 [-0.00103, +0.00018] |
| convnext_tiny | SCALE_H | +0.02341 [+0.01929, +0.02779] | +0.00013 [+0.00000, +0.00045] |
| convnext_tiny | CP_IUT_H | +0.02565 [+0.02118, +0.03029] | +0.00073 [+0.00014, +0.00155] |
| convnext_tiny | COMPOSED_CP_BETTING_H | +0.01563 [+0.01288, +0.01855] | +0.00032 [+0.00000, +0.00079] |
| convnext_tiny | GEOMETRY_DIRECT_CP | +0.00087 [-0.00050, +0.00229] | -0.00048 [-0.00112, -0.00002] |

### 인증률/ECA (arch × procedure × t × B; 분모 = 15 models × 100 replicates = 1500)

| arch | procedure | t | B | ECA 평균 | 인증률 | 인증 건수/1500 |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | RATIO | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | RATIO | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | RATIO | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | RATIO | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | RATIO | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | RATIO | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | MASS_CP | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | HOEFFDING_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | DIRECT_BETTING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | DIRECT_BETTING_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | DIRECT_BETTING_H | 0.5 | 1024 | 0.00046 | 0.0047 | 7 |
| resnet18 | DIRECT_BETTING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | DIRECT_BETTING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | DIRECT_BETTING_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | SCALE_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | SCALE_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | SCALE_H | 0.5 | 1024 | 0.00015 | 0.0013 | 2 |
| resnet18 | SCALE_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | SCALE_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | SCALE_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | CP_IUT_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | COMPOSED_CP_BETTING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | COMPOSED_CP_BETTING_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | COMPOSED_CP_BETTING_H | 0.5 | 1024 | 0.00015 | 0.0013 | 2 |
| resnet18 | COMPOSED_CP_BETTING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | COMPOSED_CP_BETTING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | COMPOSED_CP_BETTING_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| resnet18 | V1_SCALE_CP | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | V1_SCALE_CP | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | V1_SCALE_CP | 0.5 | 1024 | 0.00015 | 0.0013 | 2 |
| resnet18 | V1_SCALE_CP | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| resnet18 | V1_SCALE_CP | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| resnet18 | V1_SCALE_CP | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | RATIO | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | RATIO | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | RATIO | 0.5 | 1024 | 0.00052 | 0.0020 | 3 |
| convnext_tiny | RATIO | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | RATIO | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | RATIO | 1.0 | 1024 | 0.00011 | 0.0007 | 1 |
| convnext_tiny | MASS_CP | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | MASS_CP | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | MASS_CP | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | MASS_CP | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | MASS_CP | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | MASS_CP | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 0.5 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 0.5 | 1024 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | HOEFFDING_H | 1.0 | 1024 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | DIRECT_BETTING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | DIRECT_BETTING_H | 0.5 | 512 | 0.00279 | 0.0127 | 19 |
| convnext_tiny | DIRECT_BETTING_H | 0.5 | 1024 | 0.04440 | 0.1813 | 272 |
| convnext_tiny | DIRECT_BETTING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | DIRECT_BETTING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | DIRECT_BETTING_H | 1.0 | 1024 | 0.00011 | 0.0007 | 1 |
| convnext_tiny | SCALE_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | SCALE_H | 0.5 | 512 | 0.00183 | 0.0080 | 12 |
| convnext_tiny | SCALE_H | 0.5 | 1024 | 0.04714 | 0.1880 | 282 |
| convnext_tiny | SCALE_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | SCALE_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | SCALE_H | 1.0 | 1024 | 0.00011 | 0.0007 | 1 |
| convnext_tiny | CP_IUT_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | CP_IUT_H | 0.5 | 512 | 0.00036 | 0.0013 | 2 |
| convnext_tiny | CP_IUT_H | 0.5 | 1024 | 0.00803 | 0.0313 | 47 |
| convnext_tiny | CP_IUT_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | CP_IUT_H | 1.0 | 512 | 0.00025 | 0.0013 | 2 |
| convnext_tiny | CP_IUT_H | 1.0 | 1024 | 0.00524 | 0.0313 | 47 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 0.5 | 512 | 0.00144 | 0.0060 | 9 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 0.5 | 1024 | 0.03659 | 0.1473 | 221 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 1.0 | 512 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | COMPOSED_CP_BETTING_H | 1.0 | 1024 | 0.00258 | 0.0147 | 22 |
| convnext_tiny | V1_SCALE_CP | 0.5 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | V1_SCALE_CP | 0.5 | 512 | 0.00183 | 0.0080 | 12 |
| convnext_tiny | V1_SCALE_CP | 0.5 | 1024 | 0.04714 | 0.1880 | 282 |
| convnext_tiny | V1_SCALE_CP | 1.0 | 256 | 0.00000 | 0.0000 | 0 |
| convnext_tiny | V1_SCALE_CP | 1.0 | 512 | 0.00025 | 0.0013 | 2 |
| convnext_tiny | V1_SCALE_CP | 1.0 | 1024 | 0.00524 | 0.0313 | 47 |
