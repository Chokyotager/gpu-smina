/*
 * Serialization of boost::optional in the format of the Boost releases smina
 * was developed against (unversioned, with an explicit item_version field).
 * Newer Boost gives optional a class version of 1, which conflicts with
 * smina defaulting every class to object_serializable (parsing.h) and fails
 * the archive's compile-time versioning check.
 */

#ifndef SMINA_OPTIONAL_SERIALIZATION_H
#define SMINA_OPTIONAL_SERIALIZATION_H

// claim boost's include guard so it can never be pulled in alongside this
#define BOOST_SERIALIZATION_OPTIONAL_HPP

#include <boost/optional.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>

namespace boost {
namespace serialization {

template<class Archive, class T>
void save(Archive & ar, const boost::optional<T> & ot, const unsigned int)
{
	const bool tflag(ot);
	ar << boost::serialization::make_nvp("initialized", tflag);
	if (tflag)
	{
		const boost::serialization::item_version_type item_version(0);
		ar << BOOST_SERIALIZATION_NVP(item_version);
		ar << boost::serialization::make_nvp("value", *ot);
	}
}

template<class Archive, class T>
void load(Archive & ar, boost::optional<T> & ot, const unsigned int)
{
	bool tflag;
	ar >> boost::serialization::make_nvp("initialized", tflag);
	if (!tflag)
	{
		ot.reset();
		return;
	}
	boost::serialization::item_version_type item_version(0);
	boost::serialization::library_version_type library_version(
			ar.get_library_version());
	if (boost::serialization::library_version_type(3) < library_version)
		ar >> BOOST_SERIALIZATION_NVP(item_version);
	T t;
	ar >> boost::serialization::make_nvp("value", t);
	ot = t;
}

template<class Archive, class T>
void serialize(Archive & ar, boost::optional<T> & ot, const unsigned int version)
{
	boost::serialization::split_free(ar, ot, version);
}

} // serialization
} // boost

#endif
