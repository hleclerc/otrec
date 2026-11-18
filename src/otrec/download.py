"""Files fetched over HTTP on first use and kept, so that an example names its data where the reader sees it.

    cached_download( url, sha256 = None )        the whole file
    raw_rows( url, shape, row, dtype )           one row of each image of a raw stack, by HTTP ranges ( a slice of a
                                                 scan whose projections weigh gigabytes )

The files sit in `$OTREC_DATA`, or `~/.cache/otrec`. Without internet, copy that directory from a machine that has it."""
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np


def directory():
    return Path( os.environ.get( "OTREC_DATA" ) or Path.home() / ".cache" / "otrec" )


def cached_download( url, sha256 = None, name = None ):
    """the path of the file of `url` ( saved as `name`, default the last part of its path ), downloaded if it is not
    there yet and then checked against `sha256` when given"""
    path = directory() / ( name or url.split( "?" )[ 0 ].rstrip( "/" ).split( "/" )[ -1 ] )
    if not path.exists():
        print( f"otrec.download: { url } -> { path }", flush = True )
        _write( path, lambda part: urllib.request.urlretrieve( url, part ), sha256 )
    return path


def raw_rows( url, shape, row, dtype = "<u2", batch = 200 ):
    """`[ nb_images, width ]`: the row `row` of each image of the raw stack `url` ( `shape = ( nb_images, height, width )`,
    images one after the other, rows one after the other, no header ), fetched by ranges of bytes, `batch` per request,
    and kept ( one file per stack and row )"""
    n, height, width = shape
    size = np.dtype( dtype ).itemsize * width
    name = url.split( "?" )[ 0 ].rstrip( "/" ).split( "/" )[ -1 ] + f".row{ row }"
    path = directory() / name
    if not path.exists():
        print( f"otrec.download: row { row } of the { n } images of { url } -> { path }", flush = True )
        starts = [ ( i * height + row ) * size for i in range( n ) ]

        def fetch( part ):
            with open( part, "wb" ) as out:
                for b in range( 0, n, batch ):
                    out.write( _ranges( url, starts[ b : b + batch ], size ) )
        _write( path, fetch )
    return np.fromfile( path, dtype = dtype ).reshape( n, width )


def _ranges( url, starts, size ):
    """the bytes `[ s, s + size )` for each of `starts`, in one request ( a multipart answer when there are several )"""
    req = urllib.request.Request( url, headers = { "Range": "bytes=" + ",".join( f"{ s }-{ s + size - 1 }" for s in starts ) } )
    with urllib.request.urlopen( req ) as r:
        body, status = r.read(), r.status
    if status != 206:
        raise RuntimeError( f"otrec.download: { url } does not serve ranges of bytes" )
    if len( starts ) == 1:
        return body
    # a multipart answer: each part says its range, then a blank line, then its bytes ( read by length: servers differ
    # on the content type and on the ends of lines )
    parts, pos = { }, 0
    while ( pos := body.find( b"Content-Range: bytes ", pos ) ) >= 0:
        a, b = map( int, body[ pos + 21 : body.index( b"/", pos ) ].split( b"-" ) )
        start = min( i + len( blank ) for blank in ( b"\r\n\r\n", b"\n\n" ) if ( i := body.find( blank, pos ) ) >= 0 )
        parts[ a ], pos = body[ start : start + b - a + 1 ], start + b - a + 1
    return b"".join( parts[ s ] for s in starts )


def _write( path, fill, sha256 = None ):
    """`fill( part )` writes the file beside `path`, which it becomes once checked"""
    path.parent.mkdir( parents = True, exist_ok = True )
    part = path.with_name( path.name + ".part" )
    try:
        fill( part )
    except OSError as e:
        part.unlink( missing_ok = True )
        raise RuntimeError( f"otrec.download: { path.name } could not be downloaded ( { e } ). Without internet, copy it into "
                            f"{ path.parent } ( or $OTREC_DATA ) from a machine that has it." ) from e
    if sha256 is not None and ( got := hashlib.sha256( part.read_bytes() ).hexdigest() ) != sha256:
        part.unlink()
        raise RuntimeError( f"otrec.download: { path.name } is not the expected file ( sha256 { got } )" )
    part.rename( path )
