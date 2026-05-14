# Maintainer: Will Handley <wh260@cam.ac.uk>
pkgname=python-jaxposeidon
pkgver=0.0.2.dev0
pkgrel=1
pkgdesc="JAX-friendly port of POSEIDON's transmission-spectroscopy forward model"
arch=('any')
url="https://github.com/handley-lab/jaxPOSEIDON"
license=('BSD-3-Clause')
depends=('python' 'python-jax' 'python-numpy' 'python-scipy' 'python-h5py')
makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')
checkdepends=('python-pytest')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd jaxPOSEIDON-$pkgver
    python -m build --wheel --no-isolation
}

check() {
    cd jaxPOSEIDON-$pkgver
    # Full test suite requires POSEIDON + opacity data; skipped for AUR build.
    python -c "import jaxposeidon; print(jaxposeidon.__version__)"
}

package() {
    cd jaxPOSEIDON-$pkgver
    python -m installer --destdir="$pkgdir" dist/*.whl
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
