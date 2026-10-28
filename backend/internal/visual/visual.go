// Package visual compares two versions of a committed test screenshot and
// cuts out what changed, for the review call to describe.
//
// The pixel diff runs here, never in the model: vision models are poor at
// finding a difference between two whole screenshots, and a crop of the
// changed region costs a fraction of the tokens. Stdlib only.
package visual

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/png"

	// Registered for image.Decode: baselines are PNG, some runners write JPEG.
	_ "image/jpeg"
)

const (
	// MaxPixels refuses an image before decoding it: a decoded RGBA image is
	// four bytes a pixel, a full-page screenshot inside the 1 MiB fetch cap
	// can still decode to hundreds of MB, and this runs on the single
	// review worker. 1280 by 12,500 fits, a 5K full page does not.
	MaxPixels = 16_000_000
	// MaxEdge is the longest side sent to the model, in pixels.
	MaxEdge = 1024
	// minCrop is the smallest crop side: a one-pixel change cut to its
	// bounding box shows the model nothing it can name.
	minCrop = 256
	// margin is added around the changed region on every side.
	margin = 32
	// pixelsPerToken is the conservative image token estimate (w·h/750,
	// the dearest of the vendors' published formulas) for the fit check.
	pixelsPerToken = 750
)

// ErrTooLarge is an image over MaxPixels.
var ErrTooLarge = errors.New("image too large to compare")

// Result is one compared pair, ready to send.
type Result struct {
	// Identical: the pixels did not change (a re-encode). Nothing to send.
	Identical bool
	// Before and After are PNGs, cropped to the changed region when both
	// sides have the same size, else whole; both downscaled to MaxEdge.
	// Before is nil for an added screenshot.
	Before, After []byte
	// Cropped: Before and After show the changed region only.
	Cropped bool
	// Tokens is the estimate for Before and After together.
	Tokens int
}

// Compare decodes both versions and cuts out what changed. before is nil for
// an added screenshot.
func Compare(before, after []byte) (Result, error) {
	a, err := decode(after)
	if err != nil {
		return Result{}, fmt.Errorf("after: %w", err)
	}
	if before == nil {
		out := scale(a)
		enc, err := encode(out)
		if err != nil {
			return Result{}, err
		}
		return Result{After: enc, Tokens: tokens(out)}, nil
	}
	b, err := decode(before)
	if err != nil {
		return Result{}, fmt.Errorf("before: %w", err)
	}

	res := Result{}
	if a.Rect.Size() == b.Rect.Size() {
		changed, ok := changedRegion(b, a)
		if !ok {
			return Result{Identical: true}, nil
		}
		crop := cropRect(changed, a.Rect)
		res.Cropped = crop != a.Rect
		a = sub(a, crop)
		b = sub(b, crop)
	}
	sa, sb := scale(a), scale(b)
	if res.After, err = encode(sa); err != nil {
		return Result{}, err
	}
	if res.Before, err = encode(sb); err != nil {
		return Result{}, err
	}
	res.Tokens = tokens(sa) + tokens(sb)
	return res, nil
}

// DataURI is how an inline image goes on the wire.
func DataURI(pngBytes []byte) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
}

// Hash is the hex SHA-256 of one version, the cache key's half; "" for nil
// (an added screenshot has no before).
func Hash(b []byte) string {
	if b == nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func decode(data []byte) (*image.RGBA, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, errors.New("empty image")
	}
	if cfg.Width*cfg.Height > MaxPixels {
		return nil, ErrTooLarge
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	// An opaque PNG decodes to RGBA already: no second copy.
	if rgba, ok := img.(*image.RGBA); ok && rgba.Rect.Min == (image.Point{}) {
		return rgba, nil
	}
	out := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
	draw.Draw(out, out.Rect, img, img.Bounds().Min, draw.Src)
	return out, nil
}

// changedRegion is the bounding box of every pixel that differs, exactly: a
// tolerance would hide anti-aliasing and font drift, which is the change a
// reviewer most needs told about. ok is false when nothing differs.
func changedRegion(b, a *image.RGBA) (image.Rectangle, bool) {
	w, h := a.Rect.Dx(), a.Rect.Dy()
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := range h {
		ra := a.Pix[y*a.Stride : y*a.Stride+w*4]
		rb := b.Pix[y*b.Stride : y*b.Stride+w*4]
		if bytes.Equal(ra, rb) {
			continue
		}
		for x := range w {
			if !bytes.Equal(ra[x*4:x*4+4], rb[x*4:x*4+4]) {
				minX, maxX = min(minX, x), max(maxX, x)
			}
		}
		minY, maxY = min(minY, y), max(maxY, y)
	}
	if maxX < 0 {
		return image.Rectangle{}, false
	}
	return image.Rect(minX, minY, maxX+1, maxY+1), true
}

// cropRect grows the changed region by the margin and to at least minCrop a
// side, centred on it, inside bounds.
func cropRect(r, bounds image.Rectangle) image.Rectangle {
	r = r.Inset(-margin)
	grow := func(lo, hi, limLo, limHi int) (int, int) {
		if need := minCrop - (hi - lo); need > 0 {
			lo -= need / 2
			hi += need - need/2
		}
		if lo < limLo {
			hi += limLo - lo
			lo = limLo
		}
		if hi > limHi {
			lo -= hi - limHi
			hi = limHi
		}
		return max(lo, limLo), hi
	}
	r.Min.X, r.Max.X = grow(r.Min.X, r.Max.X, bounds.Min.X, bounds.Max.X)
	r.Min.Y, r.Max.Y = grow(r.Min.Y, r.Max.Y, bounds.Min.Y, bounds.Max.Y)
	return r.Intersect(bounds)
}

// sub copies a region out to its own image at the origin.
func sub(img *image.RGBA, r image.Rectangle) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(out, out.Rect, img, r.Min, draw.Src)
	return out
}

// scale shrinks img so its longest side is at most MaxEdge, averaging each
// destination pixel's source box: nearest-neighbour would drop the thin
// strokes of text, which is most of a UI.
func scale(img *image.RGBA) *image.RGBA {
	sw, sh := img.Rect.Dx(), img.Rect.Dy()
	long := max(sw, sh)
	if long <= MaxEdge {
		return img
	}
	dw, dh := max(1, sw*MaxEdge/long), max(1, sh*MaxEdge/long)
	out := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for dy := range dh {
		y0, y1 := dy*sh/dh, max((dy+1)*sh/dh, dy*sh/dh+1)
		for dx := range dw {
			x0, x1 := dx*sw/dw, max((dx+1)*sw/dw, dx*sw/dw+1)
			var sum [4]int
			for y := y0; y < y1; y++ {
				row := img.Pix[y*img.Stride:]
				for x := x0; x < x1; x++ {
					for c := range 4 {
						sum[c] += int(row[x*4+c])
					}
				}
			}
			n := (y1 - y0) * (x1 - x0)
			o := out.Pix[dy*out.Stride+dx*4:]
			for c := range 4 {
				o[c] = uint8(sum[c] / n) //nolint:gosec // G115: an average of uint8 values
			}
		}
	}
	return out
}

func encode(img *image.RGBA) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func tokens(img *image.RGBA) int {
	return (img.Rect.Dx()*img.Rect.Dy() + pixelsPerToken - 1) / pixelsPerToken
}
