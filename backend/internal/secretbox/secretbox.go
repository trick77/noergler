// Package secretbox seals the credentials a team pushes, under the instance
// key NOERGLER_SECRETS_KEY: AES-256-GCM, a fresh random nonce per seal.
//
// The associated data binds a ciphertext to where it is stored ("<slug>|<kind>"
// for a team credential), so a row copied to another team or kind fails to
// open instead of handing that team someone else's token.
package secretbox

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// KeySize is the key length in bytes: AES-256.
const KeySize = 32

// ErrOpen is any failure to open a sealed value: wrong key, wrong associated
// data, a damaged row. GCM cannot tell them apart and neither does this.
var ErrOpen = errors.New("cannot decrypt")

// Box seals and opens with one key. Safe for concurrent use.
type Box struct {
	aead cipher.AEAD
}

// ParseKey reads a hex key of exactly KeySize bytes.
func ParseKey(s string) ([]byte, error) {
	key, err := hex.DecodeString(strings.TrimSpace(s))
	if err != nil {
		return nil, errors.New("must be hex (openssl rand -hex 32)")
	}
	if len(key) != KeySize {
		return nil, fmt.Errorf("must be %d bytes (%d hex characters), got %d bytes", KeySize, 2*KeySize, len(key))
	}
	return key, nil
}

// New builds a box for a KeySize key.
func New(key []byte) (*Box, error) {
	if len(key) != KeySize {
		return nil, fmt.Errorf("secretbox: key must be %d bytes, got %d", KeySize, len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &Box{aead: aead}, nil
}

// Seal encrypts plaintext bound to aad.
func (b *Box) Seal(plaintext, aad []byte) (nonce, ciphertext []byte, err error) {
	nonce = make([]byte, b.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, nil, err
	}
	return nonce, b.aead.Seal(nil, nonce, plaintext, aad), nil
}

// Open decrypts what Seal produced with the same aad. Any failure is ErrOpen.
func (b *Box) Open(nonce, ciphertext, aad []byte) ([]byte, error) {
	if len(nonce) != b.aead.NonceSize() {
		return nil, ErrOpen
	}
	out, err := b.aead.Open(nil, nonce, ciphertext, aad)
	if err != nil {
		return nil, ErrOpen
	}
	return out, nil
}
