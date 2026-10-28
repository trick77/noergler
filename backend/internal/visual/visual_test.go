package visual

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func pngOf(t *testing.T, w, h int, paint func(*image.RGBA)) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	if paint != nil {
		paint(img)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func size(t *testing.T, b []byte) image.Point {
	t.Helper()
	cfg, err := png.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return image.Pt(cfg.Width, cfg.Height)
}

func TestIdenticalPixelsSendNothing(t *testing.T) {
	a := pngOf(t, 100, 80, nil)
	// Re-encoded at another compression level: other bytes, same pixels.
	img, _ := png.Decode(bytes.NewReader(a))
	var buf bytes.Buffer
	_ = (&png.Encoder{CompressionLevel: png.BestCompression}).Encode(&buf, img)

	res, err := Compare(a, buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !res.Identical || res.Before != nil || res.After != nil {
		t.Fatalf("want identical with nothing to send, got %+v", res)
	}
}

func TestOnePixelIsCroppedWithContext(t *testing.T) {
	before := pngOf(t, 2000, 1500, nil)
	after := pngOf(t, 2000, 1500, func(img *image.RGBA) { img.Set(1000, 700, color.RGBA{0, 0, 0, 255}) })

	res, err := Compare(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if res.Identical || !res.Cropped {
		t.Fatalf("want a crop, got identical=%v cropped=%v", res.Identical, res.Cropped)
	}
	// Grown to the minimum crop, not cut to the one pixel.
	if got := size(t, res.After); got != image.Pt(minCrop, minCrop) {
		t.Errorf("after crop = %v, want %dx%d", got, minCrop, minCrop)
	}
	if size(t, res.Before) != size(t, res.After) {
		t.Errorf("before and after crops differ in size")
	}
	if res.Tokens <= 0 {
		t.Errorf("tokens = %d, want an estimate", res.Tokens)
	}
}

func TestCropAtTheEdgeStaysInside(t *testing.T) {
	before := pngOf(t, 300, 200, nil)
	after := pngOf(t, 300, 200, func(img *image.RGBA) { img.Set(0, 0, color.RGBA{0, 0, 0, 255}) })
	res, err := Compare(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if got := size(t, res.After); got != image.Pt(256, 200) {
		t.Errorf("edge crop = %v, want 256x200", got)
	}
}

func TestResizedScreenshotIsSentWholeAndDownscaled(t *testing.T) {
	before := pngOf(t, 1280, 720, nil)
	after := pngOf(t, 2048, 900, nil)
	res, err := Compare(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if res.Cropped {
		t.Error("a size change cannot be cropped")
	}
	if got := size(t, res.After); got != image.Pt(1024, 450) {
		t.Errorf("after = %v, want 1024x450", got)
	}
	if got := size(t, res.Before); got != image.Pt(1024, 576) {
		t.Errorf("before = %v, want 1024x576", got)
	}
}

func TestAddedScreenshotHasNoBefore(t *testing.T) {
	res, err := Compare(nil, pngOf(t, 50, 50, nil))
	if err != nil {
		t.Fatal(err)
	}
	if res.Before != nil || res.After == nil {
		t.Fatalf("added: want after only, got before=%v after=%v", res.Before != nil, res.After != nil)
	}
}

func TestHugeImageIsRefusedBeforeDecoding(t *testing.T) {
	big := pngOf(t, 5000, 4000, nil)
	if _, err := Compare(nil, big); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
}

func TestGarbageIsAnError(t *testing.T) {
	if _, err := Compare([]byte("not a png"), pngOf(t, 10, 10, nil)); err == nil {
		t.Fatal("want a decode error")
	}
}

func TestHash(t *testing.T) {
	if Hash(nil) != "" {
		t.Error("nil hashes to empty")
	}
	if Hash([]byte("a")) == Hash([]byte("b")) || len(Hash([]byte("a"))) != 64 {
		t.Error("hash is hex sha256")
	}
}
